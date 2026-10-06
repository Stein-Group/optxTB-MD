'''
NSGA-III optimization of GFN2-xTB parameters against cluster-level (E, F) targets. -> 2 objectives!
Only water (T1 tier) is optimized; T2 and T3 tiers are excluded after the train/test split.

Loss function has two types of force term (controlled by force_term_type argument):
- 'mse': normalized mean-squared-error of forces (weighted by 1/(3*N_i) and reference norm)
- 'direction': combines ||dF|| with cosine-similarity penalty; weighting factor alpha controls
  the directional component.

User can select deterministic train/test split logic via split_dataset argument:
- If False, all structures are used for training.
- If True, structures are split by tiers and ion types in relation to their difficulty (F_term).
  For T1 and T2_* only sign=='eq' structures go to train; for T3 eq structures are allowed in
  test. In each tier approximately test_frac structures are selected for test, distributed evenly
  across F_term range.

A full-parameter L2 regularization term (Reg) is always computed. reg_as_objective controls
whether it enters the optimization as a 3rd Pareto objective (True) or as an inequality
constraint Reg <= REG_CAP (False); Reg is never dropped entirely. The regularization score
param_regularization(v) = mean_i((v_i - theta0_i)^2 / param_scale_i^2) is 0 at the GFN2
default vector and at most 1 when every parameter is at its bound. use_baseline_constraint
additionally caps E and F at their values for the default GFN2-xTB parameter vector.

Per-generation snapshots: after every generation the optimizer's own non-dominated set
(algorithm.opt, i.e. pymoo's internal n_nds solutions) is written to
OUTPUT_ROOT/RUN_ID/generation_fronts/ as both compressed .npz and .csv files, plus a
"latest_generation_front.*" pair for live inspection.

HPC additions in this version:
- command-line/environment configuration was added for RUN_ID, SCRATCH_ROOT, OUTPUT_ROOT,
  N_WORKERS, and DFTB_THREADS;
- all mutable DFTB+ files are written under a run-specific scratch directory, so independent
  Slurm jobs do not overwrite each other;
- the per-structure DFTB+ calculations are parallelized with a ThreadPoolExecutor, while each
  DFTB+ process is kept single-threaded by default;
- monomer reference energies are calculated once per parameter vector instead of once per
  structure;
- final Pareto results and run_config.json are written to OUTPUT_ROOT/RUN_ID.
'''

import numpy as np
import re, sys
import os
import argparse
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import (
    calculate_dftbplus_energy_reduced,
    calculate_e_unbiased_f_md,
    ensure_param_file,
    get_total_charge,
    calculate_re_f_md,
    optimize_dftbplus_geometry_reduced,
    read_xyz,
)
from codes.gradients_qchem import calculate_re_f_md_dft
from codes.common import (
    _env_bool,
    generate_parameter_info,
    generate_sobol_initial_population,
)

import pandas as pd
from sklearn.metrics import root_mean_squared_error

import time
np.set_printoptions(threshold=np.inf)

from collections import Counter
from sklearn.metrics import r2_score

from ase.io import read 
from scipy.stats import qmc
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# --- dftb_runtime_env.py (или просто в начале твоего файла) ---
import platform

def init_dftb_env(omp_threads=4, blas_threads=1, amd_mkl=False, bind_openmp=False):
    # Не перезаписываем, если уже заданы снаружи
    os.environ.setdefault('OMP_NUM_THREADS', str(omp_threads))
    if bind_openmp:
        os.environ.setdefault('OMP_PROC_BIND', 'close')
        os.environ.setdefault('OMP_PLACES', 'cores')
    os.environ.setdefault('OMP_MAX_ACTIVE_LEVELS', '1')

    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(blas_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(blas_threads))
    os.environ.setdefault('BLIS_NUM_THREADS', str(blas_threads))
    if platform.system() == 'Darwin':
        os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(blas_threads))

    if amd_mkl:
        os.environ.setdefault('MKL_DEBUG_CPU_TYPE', '5')

# <<< ВЫЗВАТЬ ОДИН РАЗ ПЕРЕД ЛЮБЫМИ ЗАПУСКАМИ >>>
init_dftb_env(
    omp_threads=1,          # подбери 1–4 для маленьких задач
    blas_threads=1,         # BLAS в 1 поток
    amd_mkl=False,          # set MKL_DEBUG_CPU_TYPE outside only if the node/build really needs it
    bind_openmp=False       # avoid pinning many independent one-thread DFTB+ subprocesses
)

DEFAULT_SCRATCH_ROOT = Path(os.environ.get("SCRATCH", "/tmp"))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_PATH / "optimization/T1_partial_charge_schemes/hpc_runs"
DEFAULT_MONOMER_XYZ_ROOT = PROJECT_ROOT_PATH / "structures/cluster_data/monomers"

def _default_run_id():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slurm_job_id:
        return f"nsga_t1_ef_water_{slurm_job_id}_{stamp}"
    return f"nsga_t1_ef_water_local_{stamp}"


def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Run T1 water EF NSGA-III safely on HPC.")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", _default_run_id()))
    parser.add_argument("--scratch-root", default=os.environ.get("SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT)))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--dftb-threads", type=int, default=None)
    parser.add_argument("--allow-oversubscription", action="store_true", default=_env_bool("ALLOW_OVERSUBSCRIPTION", False))
    args, _unknown = parser.parse_known_args()

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    args.dftb_threads = args.dftb_threads or int(os.environ.get("DFTB_THREADS", 1))
    env_workers = os.environ.get("N_WORKERS")
    requested_workers = args.n_workers if args.n_workers is not None else (int(env_workers) if env_workers else None)
    args.n_workers = requested_workers or max(1, slurm_cpus // args.dftb_threads)
    if args.n_workers * args.dftb_threads > slurm_cpus and not args.allow_oversubscription:
        reduced_workers = max(1, slurm_cpus // args.dftb_threads)
        print(
            f"Warning: N_WORKERS={args.n_workers} with DFTB_THREADS={args.dftb_threads} "
            f"exceeds SLURM_CPUS_PER_TASK={slurm_cpus}. Reducing N_WORKERS to {reduced_workers}."
        )
        args.n_workers = reduced_workers
    args.slurm_cpus = slurm_cpus
    return args

HPC = parse_hpc_args()
RUN_ID = HPC.run_id
SCRATCH_DIR = Path(HPC.scratch_root) / "optxTB-MD" / RUN_ID
WORKDIR = SCRATCH_DIR / "work"
DFTB_RUNS = WORKDIR / "dftb_runs"
MONOMER_RUN_DIR = WORKDIR / "monomers_dftb"
RUN_OUTPUT_DIR = Path(HPC.output_root) / RUN_ID
N_WORKERS = HPC.n_workers
DFTB_THREADS = HPC.dftb_threads

WORKDIR.mkdir(parents=True, exist_ok=True)
DFTB_RUNS.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATION_FRONTS_DIR = RUN_OUTPUT_DIR / "generation_fronts"
GENERATION_FRONTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["OMP_NUM_THREADS"] = str(DFTB_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

print(f"RUN_ID: {RUN_ID}")
print(f"Scratch workdir: {WORKDIR}")
print(f"Persistent output dir: {RUN_OUTPUT_DIR}")
print(f"DFTB+ threads per worker: {DFTB_THREADS}")
print(f"Python workers: {N_WORKERS}")

MONOMER_CACHE = {}


dataset_path = str(PROJECT_ROOT_PATH / "structures/cluster_data/dataset") # do not include folders inside, only xyz files

use_oo_target = True  # append the 12 T1 O-O hexamer-network-scaling reference structures (320-331) to the training set
OO_TARGET_DIR = dataset_path + '/oo_target_small'
OO_TARGET_DFT_DIR = OO_TARGET_DIR + '/dft_results_revpbe0'

coord_files = []
# filenames are 1_*.xyz, 2_*.xyz, ...
pattern = re.compile(r'^\d+_.*\.xyz$')
for filename in os.listdir(dataset_path):
    match = pattern.match(filename)
    if match:
        file_number = int(match.group().split('_')[0])
        #print(file_number)
        coord_files.append((file_number, filename))
# sort by file number
#coord_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
coord_files.sort(key=lambda x: x[0])
#print(coord_files)

param_changes = {
    "kpair": {
    #    "H-H": 1.0,
        "O-O": 1.0,
        "O-H": 1.0,
    },
    "H": {
    #    "gam": 0.4057710000000000,
    #    "lgam": [1.0000000000000000],
    #    "gam3": 0.0800000000000000,
        "zeff": 1.1053880000000000,
        "arep": 2.2137169999999999,
    },
    "O": {
    #    "gam": 0.4518960000000000,
    #    "lgam": [1.0000000000000000, 1.1497020000000000],
    #    "gam3": -0.0517134000000000,
        "zeff": 5.7844150000000001,
        "arep": 2.1657120000000001,
    },
}

#coord_files = coord_files[:1]

names, bounds, default_gfn2_vector, structure = generate_parameter_info(param_changes, fixed_components=None)
print("Parameter names:")
print(names)
print("Bounds:")
print(bounds)
print(f"Number of included parameters {len(names)}")

# extract dft data
print('No charge is calculated, since we are optimizing (E, F)...')

DFT_LEVEL = "revpbe0"  # revPBE0-D3(0)/aug-cc-pVTZ references only

_DFT_BASE = str(PROJECT_ROOT_PATH / "structures/cluster_data/dataset")
dft_path = _DFT_BASE + '/dft_results_revpbe0'
dft_label = "revPBE0-D3/aug-cc-pVTZ"
dft_monomer_path = str(PROJECT_ROOT_PATH / "structures/cluster_data/monomers/revpbe0")
print(f"DFT reference: {dft_label} ({dft_path})")

res_ref = []
forces_ref = []
molecular_dipoles_ref = []
for _, filename in coord_files:
    #print(f"Processing DFT reference for {filename}")
    coord_file_path = os.path.join(dataset_path, filename)
    RE_ref, F_ref, mu_ref = calculate_re_f_md_dft(coord_file_path, path=dft_path, verbose=False, path_monomers=dft_monomer_path)
    res_ref.append(RE_ref)
    forces_ref.append(np.array(F_ref))
    molecular_dipoles_ref.append(mu_ref)
res_ref = np.array(res_ref)
N_atoms_ref = np.array([np.asarray(forces_ref[i]).shape[0] for i in range(len(res_ref))], dtype=float)
if np.any(N_atoms_ref == 0):
        raise ValueError("Encountered structure with zero atoms.")


# GFN2-xTB calculator preparation
def get_monomer_energies(param_changes):
    key = json.dumps(param_changes, sort_keys=True, separators=(",", ":"))
    if key in MONOMER_CACHE:
        return MONOMER_CACHE[key]

    e_na = calculate_dftbplus_energy_reduced(
        param_changes,
        DEFAULT_MONOMER_XYZ_ROOT / "na/na.xyz",
        MONOMER_RUN_DIR / "na",
        charge=1,
    )
    e_cl = calculate_dftbplus_energy_reduced(
        param_changes,
        DEFAULT_MONOMER_XYZ_ROOT / "cl/cl.xyz",
        MONOMER_RUN_DIR / "cl",
        charge=-1,
    )
    e_water = optimize_dftbplus_geometry_reduced(
        param_changes,
        DEFAULT_MONOMER_XYZ_ROOT / "water/water.xyz",
        MONOMER_RUN_DIR / "water",
        charge=0,
    )
    MONOMER_CACHE[key] = (e_na, e_cl, e_water)
    return MONOMER_CACHE[key]

def calculate_dict(param_changes, coord_files):
    # params file is created once for all functions invloving it
    param_path = ensure_param_file(param_changes, DIR=WORKDIR)
    monomer_energies = get_monomer_energies(param_changes)

    def job(i_file):
        i, (file_number, filename) = i_file
        coord_file_path = os.path.join(dataset_path, filename)
        total_charge = get_total_charge(coord_file_path)
        run_dir = DFTB_RUNS / f"{i:05d}_{Path(filename).stem}__q{total_charge:+d}"
        RE, F, mu = calculate_re_f_md(
            param_changes,
            coord_file_path,
            total_charge,
            run_dir=run_dir,
            monomer_energies=monomer_energies,
            monomer_xyz_root=DEFAULT_MONOMER_XYZ_ROOT,
            param_path=param_path,
            verbose=False,
        )
        return i, RE, np.array(F), mu

    idx_files = list(enumerate(coord_files))
    res = [None] * len(idx_files)
    forces = [None] * len(idx_files)
    molecular_dipoles = [None] * len(idx_files)
    errors = {}

    with ThreadPoolExecutor(max_workers=min(N_WORKERS, len(idx_files))) as ex:
        futs = [ex.submit(job, item) for item in idx_files]
        for fut in as_completed(futs):
            try:
                i, RE, F, mu = fut.result()
                res[i] = RE
                forces[i] = F
                molecular_dipoles[i] = mu
            except Exception as e:
                errors[str(fut)] = repr(e)

    failed = [
        i for i in range(len(res))
        if res[i] is None or forces[i] is None or molecular_dipoles[i] is None
    ]
    if failed:
        raise RuntimeError(
            f"Some DFTB+ jobs failed: {len(failed)}/{len(res)}. "
            f"Example error: {next(iter(errors.values())) if errors else 'unknown'}"
        )

    return np.array(res), forces, molecular_dipoles

def calculate(vector, coord_files):
    param_changes = vector_to_param_changes(vector, structure)
    return calculate_dict(param_changes, coord_files)

# ----------------------- chose dataset split ----------------------- #

split_dataset = True

if split_dataset:
    print('Splitting the dataset...')
    def classify_tier_and_ion(file_number, counts):
        n_na = counts.get('Na', 0)
        n_cl = counts.get('Cl', 0)

        # сначала по диапазонам индексов — так ты гарантированно совпадёшь со своей логикой
        if 1 <= file_number <= 81:
            tier = 'T1'
            ion_type = 'none'
        elif 82 <= file_number <= 180:
            tier = 'T2_Na'
            ion_type = 'Na+'
        elif 181 <= file_number <= 289:
            tier = 'T2_Cl'
            ion_type = 'Cl-'
        elif 290 <= file_number <= 307:
            tier = 'T3'
            ion_type = 'NaCl'
        else:
            # fallback по составу
            if n_na == 0 and n_cl == 0:
                tier = 'T1'
                ion_type = 'none'
            elif n_na > 0 and n_cl == 0:
                tier = 'T2_Na'
                ion_type = 'Na+'
            elif n_na == 0 and n_cl > 0:
                tier = 'T2_Cl'
                ion_type = 'Cl-'
            else:
                tier = 'T3'
                ion_type = 'NaCl'

        return tier, ion_type


    def parse_isomer_and_distortion(filename_base):
        """
        filename_base — часть имени БЕЗ ведущего номера и .xyz.
        Например:
        'h2o_mode01_minus'  -> isomer_id='h2o',      distortion_id='mode01_minus'
        'H2O_2'             -> isomer_id='H2O_2',    distortion_id=None
        'Na_h2o3_mode02_plus' -> isomer_id='Na_h2o3', distortion_id='mode02_plus'
        """
        base = filename_base  # уже без номера и .xyz

        if 'mode' in base:
            prefix, mode_part = base.split('mode', 1)
            isomer_id = prefix.rstrip('_')
            m = re.match(r'(\d+)_?(minus|plus)', mode_part)
            if m: 
                mode_id = int(m.group(1))
                sign = m.group(2)
            else:
                mode_id = None
                sign = None
            #distortion_id = 'mode' + mode_part
        else:
            isomer_id = base
            mode_id = None
            sign = 'eq' # original geometry, undistorted
            #distortion_id = None

        return isomer_id.lower(), mode_id, sign

    metadata = []

    for file_number, filename in coord_files:
        coord_file_path = os.path.join(dataset_path, filename)
        _, symbols, _ = read_xyz(coord_file_path)
        counts = Counter(symbols)
        n_atoms = len(symbols)
        n_O = counts.get('O', 0)
        n_H = counts.get('H', 0)
        n_Na = counts.get('Na', 0)
        n_Cl = counts.get('Cl', 0)
        n_waters = n_O
        total_charge = get_total_charge(coord_file_path)
        tier, ion_type = classify_tier_and_ion(file_number, counts)
        #print(f"File: {filename}, Tier: {tier}, Ion type: {ion_type}, Total charge: {total_charge}")
        base_part = filename.split('_', 1)[1].rsplit('.', 1)[0]
        isomer_id, mode_id, sign = parse_isomer_and_distortion(base_part)
        group_id = (tier, ion_type, n_waters, isomer_id)
        #print(f"File: {filename}, Group ID: {group_id}")
        metadata.append(
            dict(
                idx = file_number,
                filename = filename,
                tier = tier,
                ion_type = ion_type,
                group_id = group_id,
                n_atoms = n_atoms,
                n_waters = n_waters,
                n_O = n_O,
                n_H = n_H,
                n_Na = n_Na,
                n_Cl = n_Cl,
                total_charge = total_charge,
                isomer_id = isomer_id,
                mode_id = mode_id,
                sign = sign,
            )
        )
        
    meta_df = pd.DataFrame(metadata).set_index("idx").sort_index()

    def evaluate_f_term_single(vector, coordinate_idx, F_denominator, coord_files, refs, penalty_value=1e2, eps=1e-12):
        # reference data
        F_ref_list = refs[1][coordinate_idx]

        # model predictions with current params
        try:
            _, F_pred_list, _ = calculate(vector, [coord_files[coordinate_idx]])
        except Exception as e:
            print(f"Exception during model calculation: {e}")
            return penalty_value
        
        # force term - а это точно что было в статье?
        F_numerator = 0.0
        Fref = np.asarray(F_ref_list, dtype=float)
        Fpred = np.asarray(F_pred_list[0], dtype=float)
        if Fref.shape != Fpred.shape or Fref.ndim != 2 or Fref.shape[1] != 3:
            print(f"Inconsistent force shapes encountered: {Fref.shape} vs {Fpred.shape}.")
            return penalty_value
        Ni = Fref.shape[0]
        dF = Fpred - Fref
        wF_i = 1 / (3 * Ni)
        F_numerator += wF_i * float(np.sum(dF * dF))
        
        F_term = F_numerator / (F_denominator + eps)
        L = F_term
        if not np.isfinite(L):
            print("Non-finite loss encountered.")
            L = penalty_value
        return float(L)

    def make_f_measure(coord_files=coord_files, res_ref=res_ref, forces_ref=forces_ref):
        per_structure_F = {}
        #F_denominator = evaluate_f_denominator(coord_files=coord_files, refs=(res_ref, forces_ref, molecular_dipoles_ref))
        for element in enumerate(coord_files):
            i = int(element[1][0]) 
            j = int(element[0])
            per_structure_F[i] = evaluate_f_term_single(default_gfn2_vector, j, 1.0, coord_files=coord_files, refs=(res_ref, forces_ref))
            #print(f"Structure {element[1][1]}: F_term = {per_structure_F[i]:.10f}")
        return per_structure_F

    per_structure_F = make_f_measure(coord_files=coord_files, res_ref=res_ref, forces_ref=forces_ref)

    meta_df["F_term"] = meta_df.index.map(lambda i: per_structure_F.get(i, np.nan))

    def split_train_test(
        meta_df,
        test_frac=0.2,
        eq_to_train_tiers=("T1", "T2_Na", "T2_Cl"),
        tiers_order=("T1", "T2_Na", "T2_Cl"),
        add_T3 = True,
    ):
        """
        Делит весь датасет на train/test.

        Логика:
        - для T1 и T2_* все sign=='eq' -> в train;
        - для T3 eq разрешены в test;
        - в каждом tier выбираем примерно test_frac структур в тест,
        распределяя их равномерно по сложности (F_term).
        """

        meta = meta_df.copy()

        # Подготовим F_term: заполним NaN медианой, чтобы можно было сортировать
        if "F_term" not in meta.columns:
            raise ValueError("meta_df must contain 'F_term' column with per-structure difficulty.")
        meta["F_term_eff"] = meta["F_term"]
        if meta["F_term_eff"].isna().any():
            print("Warning: NaN values found in 'F_term' column; filling with median.")
            #median_F = meta["F_term_eff"].median()
            #meta["F_term_eff"] = meta["F_term_eff"].fillna(median_F)

        train_indices = []
        test_indices = []

        for tier in tiers_order:
            tier_mask = (meta["tier"] == tier)
            tier_idx = meta.index[tier_mask].to_numpy()
            if len(tier_idx) == 0:
                continue  # такого tier может не быть

            # Какие индексы считать "eq"
            is_eq = (meta.loc[tier_idx, "sign"] == "eq")

            # В каких tier eq нельзя отправлять в test
            if tier in eq_to_train_tiers:
                eq_idx = tier_idx[is_eq.to_numpy()]
                candidate_idx = tier_idx[~is_eq.to_numpy()]  # только деформированные
            else:
                eq_idx = np.array([], dtype=int)
                candidate_idx = tier_idx  # для T3 все кандидаты

            # Целевое количество структур в тесте внутри этого tier
            n_tier = len(tier_idx)
            n_test_target = int(round(test_frac * n_tier))
            #print(f"Tier {tier}: total={n_tier}, target test={n_test_target}, eq in train={len(eq_idx)}, candidates={len(candidate_idx)}")

            # Гарантия, что число адекватно
            # (минимум 1, максимум число кандидатов; если кандидатов мало - все идут в test)
            n_test_target = max(1, n_test_target)
            n_test_target = min(n_test_target, len(candidate_idx))

            if len(candidate_idx) == 0 or n_test_target <= 0:
                # Некого отправлять в test (например, в tier только eq, а мы их запрещаем)
                # Тогда просто всё в train
                train_indices.extend(list(tier_idx))
                continue

            # Сортируем кандидатов по F_term (сложности)
            cand_df = meta.loc[candidate_idx].sort_values("F_term_eff")
            cand_sorted = cand_df.index.to_numpy()

            # Выбираем n_test_target позиций равномерно по диапазону сложностей
            if n_test_target >= len(cand_sorted):
                chosen_test = cand_sorted
            else:
                positions = np.linspace(
                    0, len(cand_sorted) - 1, num=n_test_target, dtype=int
                )
                positions = np.unique(positions)  # на всякий случай
                chosen_test = cand_sorted[positions]

            # Записываем test для этого tier
            test_indices.extend(list(chosen_test))

            # Всё остальное, что не попало в test, идёт в train
            tier_train = np.setdiff1d(tier_idx, chosen_test)
            train_indices.extend(list(tier_train))

        # Убираем дубликаты и сортируем
        train_indices = np.array(sorted(set(train_indices)))
        test_indices = np.array(sorted(set(test_indices)))

        # На всякий случай гарантируем, что нет пересечения
        inter = np.intersect1d(train_indices, test_indices)
        if len(inter) > 0:
            raise RuntimeError(f"Train/Test overlap in indices: {inter}")

        # add T3 manually
        # let's add 2 CIPs in all sets
        if add_T3:
            train_indices = np.concatenate([train_indices, [290, 291, 293, 294]])
            test_indices = np.concatenate([test_indices, [292, 295]])
            # not CIP: split such that r_nacl coverage is maximized, also include 2nd solvation shell several times
            train_indices = np.concatenate([train_indices, [296, 298, 299, 301, 304, 307]])
            test_indices = np.concatenate([test_indices, [297, 300, 302, 303, 305, 306]])


        return train_indices, test_indices
    
    train_indices, test_indices = split_train_test(meta_df, test_frac=0.25, add_T3=True)
    print(f'Split sizes: Train={len(train_indices)}, Test={len(test_indices)}, Overall={len(train_indices)+len(test_indices)}')
    train_coord = [coord_files[i] for i in train_indices - 1]
    train_refs = (np.asarray(res_ref)[train_indices - 1], [forces_ref[i] for i in train_indices - 1])
    train_N_atoms = np.asarray(N_atoms_ref)[train_indices - 1]
    test_coord = [coord_files[i] for i in test_indices - 1]
    test_refs = (np.asarray(res_ref)[test_indices - 1], [forces_ref[i] for i in test_indices - 1])
    test_N_atoms = np.asarray(N_atoms_ref)[test_indices - 1]

    # select only tier 1 in the train and test sets
    train_mask = np.isin(meta_df.loc[train_indices, "tier"], ["T1"])
    test_mask = np.isin(meta_df.loc[test_indices, "tier"], ["T1"])
    train_coord = [train_coord[i] for i in range(len(train_coord)) if train_mask[i]]
    train_refs = (np.asarray(train_refs[0])[train_mask], [train_refs[1][i] for i in range(len(train_refs[1])) if train_mask[i]])
    train_N_atoms = np.asarray(train_N_atoms)[train_mask]
    test_coord = [test_coord[i] for i in range(len(test_coord)) if test_mask[i]]
    test_refs = (np.asarray(test_refs[0])[test_mask], [test_refs[1][i] for i in range(len(test_refs[1])) if test_mask[i]])
    test_N_atoms = np.asarray(test_N_atoms)[test_mask]

else:
    print('Preparing the dataset (T1)...')
    # select only tier1 in the train set, i.e. remove everything apart 81 structures
    train_coord = coord_files[:81]
    train_refs = ([res_ref[i] for i in range(81)], [forces_ref[i] for i in range(81)])
    train_N_atoms = N_atoms_ref[:81]
    #print('Preparing the dataset (all_tiers)...')
    #train_coord = coord_files
    #train_refs = (np.asarray(res_ref), forces_ref, charges_ref)
    #train_N_atoms = N_atoms_ref
    test_coord = None
    test_refs = None
    test_N_atoms = None

if use_oo_target:
    oo_target_files = [(int(pattern.match(f).group().split('_')[0]), f)
                        for f in sorted(os.listdir(OO_TARGET_DIR)) if pattern.match(f)]
    print(f"use_oo_target=True: adding {len(oo_target_files)} O-O target structures: {[f for _, f in oo_target_files]}")
    oo_res, oo_forces = [], []
    for _, filename in oo_target_files:
        coord_file_path = os.path.join(OO_TARGET_DIR, filename)
        RE, F, _ = calculate_re_f_md_dft(coord_file_path, path=OO_TARGET_DFT_DIR, verbose=False, path_monomers=dft_monomer_path)
        oo_res.append(RE)
        oo_forces.append(np.array(F))
    train_coord = train_coord + [(n, f"oo_target_small/{f}") for n, f in oo_target_files]
    train_refs = (np.concatenate([train_refs[0], np.array(oo_res)]), list(train_refs[1]) + oo_forces)
    train_N_atoms = np.concatenate([train_N_atoms, np.array([len(f) for f in oo_forces], dtype=float)])

print(f'Number of training structures: {len(train_coord)}')

run_config = {
    "run_id": RUN_ID,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_cpus_per_task": HPC.slurm_cpus,
    "n_workers": N_WORKERS,
    "dftb_threads": DFTB_THREADS,
    "scratch_dir": str(SCRATCH_DIR),
    "workdir": str(WORKDIR),
    "dftb_runs": str(DFTB_RUNS),
    "monomer_run_dir": str(MONOMER_RUN_DIR),
    "output_dir": str(RUN_OUTPUT_DIR),
    "dataset_path": str(dataset_path),
    "use_oo_target": use_oo_target,
    "monomer_xyz_root": str(DEFAULT_MONOMER_XYZ_ROOT),
    "n_train_structures": len(train_coord),
    "n_test_structures": 0 if test_coord is None else len(test_coord),
    "dft_level": DFT_LEVEL,
    "dft_monomer_path": dft_monomer_path,
    "dft_label": dft_label,
}
(RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
print(f"Run config saved to {RUN_OUTPUT_DIR / 'run_config.json'}")

bounds_arr = np.asarray(bounds, dtype=float)
lower = bounds_arr[:, 0]
upper = bounds_arr[:, 1]

# typical parameter scale: half-width of the bounds
param_scale = 0.5 * (upper - lower)
# avoid zero to prevent division by zero
param_scale[param_scale == 0.0] = 1.0

theta0 = np.asarray(default_gfn2_vector, dtype=float)

# Reg (parameter regularization) is always computed; this toggle controls only HOW it enters
# the optimization: as a 3rd Pareto objective (True) or as an inequality constraint
# Reg <= REG_CAP (False). Reg is never dropped entirely.
reg_as_objective = True
REG_CAP = 0.6  # cap used only when reg_as_objective=False

# If True, additionally constrains E <= E0 and F <= F0 (loss components at the default
# GFN2-xTB parameters). Never applied to Reg (Reg0 is trivially 0 at the default vector).
# Default False so reg_as_objective=True reproduces the pre-existing results exactly.
use_baseline_constraint = False

objective_names = ["E", "F", "Reg"] if reg_as_objective else ["E", "F"]

def param_regularization(vector, theta0=theta0, scale=param_scale):
    """Normalized L2 deviation of all parameters from their default values.

    Returns mean_i((theta_i - theta0_i)^2 / param_scale_i^2), where
    param_scale_i = 0.5*(upper_i - lower_i). Value is 0 at the default
    point and at most 1 when every parameter is at its bound.
    """
    v = np.asarray(vector, dtype=float)
    diff = (v - theta0) / scale
    return float(np.mean(diff * diff))

wE=0.1
wF=0.9
alpha = 2.0
print(f"Loss component weights for diagnostic L only: wE={wE}, wF={wF}, alpha={alpha}")
print(f"Reg as objective: {reg_as_objective} (reg_cap={REG_CAP}), use_baseline_constraint: {use_baseline_constraint}")

def loss(vector, coord_files=train_coord, wE=wE, wF=wF, refs=train_refs, N_atoms=train_N_atoms, penalty_value=1e6, eps=1e-12, return_components=False, print_components=False, force_term_type="mse", alpha=alpha):
    """
    Normalized objective:
        L = wE * [ Σ_i (1/N_i) ( (ΔE_i - <ΔE>)^2 ) / Σ_i (1/N_i) (E_ref_i - <E_ref>)^2 ]
          + wF * [ Σ_i (1/(3N_i)) ||ΔF_i||^2 / Σ_i (1/(3N_i)) ||F_ref_i||^2 ]
    or directional force term when selected.

    Where:
      - ΔE_i = RE_pred_i - RE_ref_i   (RE = your fragment-referenced relative energy)
      - N_i = number of atoms in structure i
      - ||·||^2 sums over all 3*Ni force components
      - <·> are weighted means using weights 1/N_i (to mirror the sums)
    """
    #print(vector)
    # reference data
    E_ref = np.asarray(refs[0], dtype=float)                  # shape: (n_struct,)
    F_ref_list = refs[1]                                      # list of (Ni,3)

    # model predictions with current params
    try:
        E_pred, F_pred_list, _ = calculate(vector, coord_files)
    except Exception as e:
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
    
    # something went wrong in the calcs, but we did not get it in the calculate()
    if E_pred is None or F_pred_list is None:
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
    n = len(E_ref)
    if not (len(E_pred) == len(F_ref_list) == len(F_pred_list) == n):
        #raise ValueError("Reference/prediction list lengths are inconsistent.")
        # something in model prediction went wrong?
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})

    # energy term
    #dE = (np.asarray(E_pred, dtype=float) - E_ref) / N_atoms
    dE = np.asarray(E_pred, dtype=float) - E_ref
    wE_i = 1 / N_atoms  # shape: (n_struct,)
    #wE_i = 1
    sum_wE = np.sum(wE_i)
    mean_dE = np.sum(wE_i * dE) / sum_wE
    mean_Eref = np.sum(wE_i * E_ref) / sum_wE
    #print(sum_wE)
    E_numerator = np.sum(wE_i * (dE - mean_dE) ** 2)
    E_denominator = np.sum(wE_i * (E_ref - mean_Eref) ** 2) 

    # force term - а это точно что было в статье?
    F_numerator = 0.0
    F_denominator = 0.0
    for i in range(n):
        Fref = np.asarray(F_ref_list[i], dtype=float)  # shape: (Ni,3)
        Fpred = np.asarray(F_pred_list[i], dtype=float)
        # if shapes are inconsistent, something went wrong for xTB, return penalty
        if Fref.shape != Fpred.shape or Fref.ndim != 2 or Fref.shape[1] != 3:
            return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
        Ni = Fref.shape[0]
        if force_term_type == 'mse':
            dF = Fpred - Fref
            wF_i = 1 / (3 * Ni)
            F_numerator += wF_i * float(np.sum(dF * dF))
            F_denominator += wF_i * float(np.sum(Fref * Fref))
        elif force_term_type == 'direction':
            dF = Fpred - Fref
            dF_norm = np.linalg.norm(dF, axis=1)  # shape: (Ni,)
            Fref_norm = np.linalg.norm(Fref, axis=1)  # shape: (Ni,)
            Fpred_norm = np.linalg.norm(Fpred, axis=1)  # shape: (Ni,)
            cos_theta = np.sum(Fpred * Fref, axis=1) / (Fpred_norm * Fref_norm + eps) 
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
            directional_factor = 1.0 + alpha * (1.0 - cos_theta)  # 1 when perfectly aligned, increases as angle grows
            wF_i = 1 / (3 * Ni)
            F_numerator += wF_i * float(np.sum(directional_factor * dF_norm))
            F_denominator += wF_i * float(np.sum(Fref_norm))
        else:
            raise ValueError(f"Unknown force_term_type: {force_term_type}. Use 'mse' or 'direction'.")


    # stabilize denominators; if any are effectively zero, penalize (dataset is degenerate)
    if not np.isfinite(E_denominator) or E_denominator < eps:
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
    if not np.isfinite(F_denominator) or F_denominator < eps:
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
    
    E_term = E_numerator / (E_denominator + eps)
    F_term = F_numerator / (F_denominator + eps)
    Reg_term = param_regularization(vector)
    L = wE * E_term + wF * F_term
    if not np.isfinite(L):
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
    if print_components:
        print(f"Loss: {L:.10f} (dE: {E_term:.10f}, dF: {F_term:.10f}, Reg: {Reg_term:.10f})")
    if return_components:
        return float(L), {"E": float(E_term), "F": float(F_term), "Reg": float(Reg_term)}
    return float(L)

def plot_initial_parity(coord_files, refs, set_name):
    """Save force and energy parity plots for default GFN2 vs DFT reference."""
    hartree_to_mev = 27.2114 * 1000
    bohr_to_ang = 0.529177
    conv = hartree_to_mev / bohr_to_ang
    hartree_to_kcalmol = 627.503

    try:
        E_pred_list, F_pred_list, _ = calculate(default_gfn2_vector, coord_files)
    except Exception as e:
        print(f"plot_initial_parity: calculate() failed for {set_name}: {e}")
        return

    F_pred_list_conv = [F * conv for F in F_pred_list]
    F_ref_list_conv = [refs[1][i] * conv for i in range(len(refs[1]))]

    # --- force component parity (x, y, z) ---
    F_ref_all = np.concatenate([F_ref_list_conv[i] for i in range(len(F_ref_list_conv))], axis=0).ravel()
    F_pred_all = np.concatenate([F_pred_list_conv[i] for i in range(len(F_pred_list_conv))], axis=0).ravel()
    print(f"[{set_name}] Force RMSE (GFN2 default): {root_mean_squared_error(F_ref_all, F_pred_all):.6f} meV/Å")

    component_names = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=False, sharey=False)
    for comp, ax in enumerate(axes):
        F_ref = np.concatenate([F_ref_list_conv[i][:, comp] for i in range(len(F_ref_list_conv))])
        F_pred = np.concatenate([F_pred_list_conv[i][:, comp] for i in range(len(F_pred_list_conv))])
        ax.scatter(
            F_ref, F_pred, alpha=0.5, marker="o",
            label="GFN2-xTB, R$^2$={:.4f}, RMSE={:.2f} meV/Å".format(
                r2_score(F_ref, F_pred),
                root_mean_squared_error(F_ref, F_pred),
            ),
        )
        vmin = min(F_ref.min(), F_pred.min())
        vmax = max(F_ref.max(), F_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r--")
        ax.set_xlabel(f"Reference Forces ({dft_label}) (meV/Å)", fontsize=12)
        if comp == 0:
            ax.set_ylabel("Predicted Forces (meV/Å)", fontsize=12)
        ax.set_title(f"Component {component_names[comp]}")
        ax.grid(True)
        ax.legend(fontsize=10)
    fig.suptitle(f"Force component parity plots ({set_name} set) - GFN2 default", y=0.98, fontsize=12)
    plt.tight_layout()
    out_forces = RUN_OUTPUT_DIR / f"initial_parity_{set_name}_forces.png"
    plt.savefig(out_forces, dpi=150)
    plt.close()
    print(f"Saved {out_forces}")

    # --- directional deviation (GFN2 only) ---
    scalar_product = []
    magnitude_pred = []
    for struc in range(len(coord_files)):
        F_ref = np.asarray(F_ref_list_conv[struc], dtype=float)
        F_pred = np.asarray(F_pred_list_conv[struc], dtype=float)
        for atom in range(len(F_ref)):
            mag_ref = np.linalg.norm(F_ref[atom])
            mag_pred = np.linalg.norm(F_pred[atom])
            unit_ref = F_ref[atom] / (mag_ref + 1e-12)
            unit_pred = F_pred[atom] / (mag_pred + 1e-12)
            scalar_product.append(1 - np.dot(unit_ref, unit_pred))
            magnitude_pred.append(mag_pred)
    scalar_product = np.array(scalar_product)
    magnitude_pred = np.array(magnitude_pred)
    smin = np.amin(scalar_product * magnitude_pred)
    smax = np.amax(scalar_product * magnitude_pred)
    print(f"[{set_name}] GFN2 directional deviation: min={smin:.6f}, max={smax:.6f}, mean={np.mean(scalar_product * magnitude_pred):.6f}, median={np.median(scalar_product * magnitude_pred):.6f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(np.arange(len(scalar_product)), scalar_product * magnitude_pred, alpha=0.5, color='tab:blue',
               label=f'Mean {np.mean(scalar_product * magnitude_pred):.3f}, median {np.median(scalar_product * magnitude_pred):.3f} meV/Å')
    ax.set_title(f'GFN2: Directional Deviation ({set_name} set)')
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('[1 - (scalar product of force unit vectors)]*(force magnitude per atom)', fontsize=9)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    out_dir = RUN_OUTPUT_DIR / f"initial_parity_{set_name}_directions.png"
    plt.savefig(out_dir, dpi=150)
    plt.close()
    print(f"Saved {out_dir}")

    # --- energy scatter ---
    E_pred_conv = np.array(E_pred_list) * hartree_to_kcalmol
    E_ref_conv = np.array(refs[0]) * hartree_to_kcalmol
    print(f"[{set_name}] Energy RMSE (GFN2 default): {root_mean_squared_error(E_ref_conv, E_pred_conv):.3f} kcal/mol")

    plt.figure(figsize=(5, 5))
    plt.scatter(E_ref_conv, E_pred_conv, alpha=0.5, marker='o',
                label='GFN2-xTB, R^2 = {:.4f}, \nRMSE = {:.3f} kcal/mol'.format(
                    r2_score(E_ref_conv, E_pred_conv),
                    root_mean_squared_error(E_ref_conv, E_pred_conv)))
    plt.xlabel(f"Reference Energies ({dft_label}) (kcal/mol)")
    plt.ylabel("Predicted Energies (kcal/mol)")
    plt.title(f"Scatter plot of Energies ({set_name} set) - GFN2 default")
    plt.plot([E_ref_conv.min(), E_ref_conv.max()], [E_ref_conv.min(), E_ref_conv.max()], 'r--', label='y=x')
    plt.grid()
    plt.legend()
    out_ene = RUN_OUTPUT_DIR / f"initial_parity_{set_name}_energies.png"
    plt.savefig(out_ene, dpi=150)
    plt.close()
    print(f"Saved {out_ene}")


calculate_L = True

force_term_type = "mse"  # "mse" or "direction"
print(f"Selected force term type: {force_term_type}")

E0 = F0 = None  # baseline loss components at default GFN2-xTB params; used only if use_baseline_constraint=True

if calculate_L:
    print('Running test loss calculation...')
    start_time = time.time()
    #L = loss(default_gfn2_vector, coord_files, return_components=True, wE=0.2, wF=0.4, wMu=0.4, lambda_reg=0.01)
    L, baseline_components = loss(default_gfn2_vector, print_components=True, force_term_type=force_term_type, alpha=2.0, return_components=True)
    end_time = time.time()
    sweep_time = end_time - start_time
    print(f"Sweep time: {sweep_time:.2f} seconds")
    print(L)
    E0, F0 = baseline_components["E"], baseline_components["F"]
    print(f"Baseline caps (used only if use_baseline_constraint=True): E0={E0:.10f}, F0={F0:.10f}")
    print('Generating initial parity plots...')
    plot_initial_parity(train_coord, train_refs, "train")
    if test_coord is not None:
        plot_initial_parity(test_coord, test_refs, "test")

if use_baseline_constraint and (E0 is None or F0 is None):
    raise RuntimeError("use_baseline_constraint=True requires calculate_L=True to compute baseline E0/F0.")

# trying to correct loss function for pymoo
last_loss_x = None
last_loss_result = None

def evaluate_components_for_pymoo(vector, coord_files=train_coord, refs=train_refs, N_atoms=train_N_atoms, penalty_value=1e6, eps=1e-12, force_term_type=force_term_type, alpha=alpha):
    """
    Calling previous loss(..., return_components=True) ONCE and returns: L, E_term, F_term, Reg_term.
    Adding 'is_dead' to mark the point which is unusable for Pareto search (convergence problems).
    """
    global last_loss_x, last_loss_result

    x = np.asarray(vector, dtype=float)

    # if x is the same as in previous call, return cached results
    if last_loss_x is not None and np.array_equal(x, last_loss_x):
        return last_loss_result

    L, components = loss(x, coord_files=coord_files, refs=refs, N_atoms=N_atoms, penalty_value=penalty_value, eps=eps, return_components=True, force_term_type=force_term_type, alpha=alpha)

    E_term   = float(components["E"])
    F_term   = float(components["F"])
    Reg_term = float(components["Reg"])

    vals = np.array([L, E_term, F_term, Reg_term], dtype=float)

    # dead point penalty
    is_dead = (
        (not np.all(np.isfinite(vals))) or (L >= penalty_value) or (E_term >= penalty_value) or (F_term >= penalty_value)
    )
    # make dead points consistently penalized in all components, so that they are clearly dominated in Pareto sense
    if is_dead:
        L = penalty_value
        E_term = F_term = penalty_value
        Reg_term = penalty_value

    result = (float(L), float(E_term), float(F_term), float(Reg_term), bool(is_dead))

    last_loss_x = x.copy()
    last_loss_result = result

    return result

# NSGA-III optimization: add one inequality constraint to exclude non-finite points
class GFN2NSGA3Problem(ElementwiseProblem):
    def __init__(
        self,
        lower,
        upper,
        coord_files=train_coord,
        refs=train_refs,
        N_atoms=train_N_atoms,
        penalty_value=1e6,
        eps=1e-12,
        reg_as_objective=reg_as_objective,
        use_baseline_constraint=use_baseline_constraint,
        reg_cap=REG_CAP,
        E0=E0,
        F0=F0,
        force_term_type=force_term_type,
        alpha=alpha,
    ):
        n_ieq_constr = 1  # feasibility (is_dead)
        if not reg_as_objective:
            n_ieq_constr += 1  # Reg <= reg_cap
        if use_baseline_constraint:
            n_ieq_constr += 2  # E <= E0, F <= F0 (never applied to Reg; Reg0 is trivially 0)

        super().__init__(
            n_var=len(lower),
            n_obj=3 if reg_as_objective else 2,
            n_ieq_constr=n_ieq_constr,
            xl=np.asarray(lower, dtype=float),
            xu=np.asarray(upper, dtype=float),
        )

        self.coord_files = coord_files
        self.refs = refs
        self.N_atoms = N_atoms
        self.penalty_value = penalty_value
        self.eps = eps
        self.reg_as_objective = reg_as_objective
        self.use_baseline_constraint = use_baseline_constraint
        self.reg_cap = reg_cap
        self.E0 = E0
        self.F0 = F0
        self.force_term_type = force_term_type
        self.alpha = alpha

    def _evaluate(self, x, out, *args, **kwargs):
        L, E_term, F_term, Reg_term, is_dead = evaluate_components_for_pymoo(
            x,
            coord_files=self.coord_files,
            refs=self.refs,
            N_atoms=self.N_atoms,
            penalty_value=self.penalty_value,
            eps=self.eps,
            force_term_type=self.force_term_type,
            alpha=self.alpha,
        )

        if self.reg_as_objective:
            out["F"] = np.array([E_term, F_term, Reg_term], dtype=float)
        else:
            out["F"] = np.array([E_term, F_term], dtype=float)

        # Constraints: dead points are infeasible, plus optional Reg cap / baseline caps
        g_vals = [1.0 if is_dead else -1.0]
        if not self.reg_as_objective:
            g_vals.append(Reg_term - self.reg_cap)
        if self.use_baseline_constraint:
            g_vals.append(E_term - self.E0)
            g_vals.append(F_term - self.F0)
        out["G"] = np.array(g_vals, dtype=float)

        # Optional diagnostics
        out["L"] = float(L)
        out["Reg"] = float(Reg_term)


def write_population_csv(path, X, F, CV=None, extra_columns=None):
    fields = list(objective_names) + [f"param_{name}" for name in names]
    if extra_columns:
        fields.extend(extra_columns.keys())
    if CV is not None:
        fields.append("CV")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(X)):
            row = {}
            for j, objective_name in enumerate(objective_names):
                row[objective_name] = float(F[i, j])
            for j, parameter_name in enumerate(names):
                row[f"param_{parameter_name}"] = float(X[i, j])
            if extra_columns:
                for col_name, col_values in extra_columns.items():
                    row[col_name] = float(col_values[i])
            if CV is not None:
                cv_i = np.asarray(CV[i], dtype=float)
                row["CV"] = float(np.max(cv_i)) if cv_i.ndim > 0 else float(cv_i)
            writer.writerow(row)


def save_generation_front(generation, X, F, CV=None, prefix="generation"):
    out_npz = GENERATION_FRONTS_DIR / f"{prefix}_{generation:04d}.npz"
    out_csv = GENERATION_FRONTS_DIR / f"{prefix}_{generation:04d}.csv"
    np.savez_compressed(
        out_npz,
        X=X,
        F=F,
        CV=np.asarray([]) if CV is None else CV,
        parameter_names=np.asarray(names),
        objective_names=np.asarray(objective_names),
    )
    write_population_csv(out_csv, X, F, CV=CV)
    np.savez_compressed(
        GENERATION_FRONTS_DIR / "latest_generation_front.npz",
        X=X,
        F=F,
        CV=np.asarray([]) if CV is None else CV,
        generation=int(generation),
        parameter_names=np.asarray(names),
        objective_names=np.asarray(objective_names),
    )
    write_population_csv(GENERATION_FRONTS_DIR / "latest_generation_front.csv", X, F, CV=CV)


class GenerationFrontCallback(Callback):
    """Save algorithm.opt (pymoo's own non-dominated set) after each generation.

    Unlike the PBC variant which recomputes the Pareto front manually from algorithm.pop,
    here we use algorithm.opt directly so the saved population exactly matches pymoo's
    internal n_nds count.
    """

    def notify(self, algorithm):
        generation = int(getattr(algorithm, "n_gen", 0))
        opt = algorithm.opt
        if opt is None or len(opt) == 0:
            return
        X_opt = opt.get("X")
        F_opt = opt.get("F")
        CV_opt = opt.get("CV")
        if X_opt is None or F_opt is None:
            return
        save_generation_front(generation, X_opt, F_opt, CV=CV_opt)
        print(f"Saved generation {generation} optimal population (n_nds={len(X_opt)}).")



# simulation settings
direction_number = 512 + 1  # same as pop_size
number_of_objectives = 3 if reg_as_objective else 2
algo_name = "nsga3" if number_of_objectives == 3 else "nsga2"
print(f"Number of objectives: {number_of_objectives}")
print(f"Algorithm: {algo_name.upper()}")
n_gen = 100
print(f"Number of generations: {n_gen}")

mutation_prob = 1.0 / len(lower) #

if algo_name == "nsga3":
    ref_dirs = get_reference_directions('energy', number_of_objectives, direction_number, seed=42)
    pop_size = len(ref_dirs)  # should be equal to direction_number
else:
    ref_dirs = None
    pop_size = direction_number
print(f"Population size: {pop_size}")

X_init = generate_sobol_initial_population(bounds, pop_size, default_gfn2_vector, seed=42, scramble=True)

problem = GFN2NSGA3Problem(
    lower=lower,
    upper=upper,
    coord_files=train_coord,
    refs=train_refs,
    N_atoms=train_N_atoms,
    penalty_value=1e6,
    eps=1e-12,
    reg_as_objective=reg_as_objective,
    use_baseline_constraint=use_baseline_constraint,
    reg_cap=REG_CAP,
    E0=E0,
    F0=F0,
    force_term_type=force_term_type,
    alpha=alpha,
)

if algo_name == "nsga3":
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=X_init, crossover=SBX(prob=0.9, eta=5), mutation=PM(prob=mutation_prob, eta=5))
else:
    algorithm = NSGA2(pop_size=pop_size, sampling=X_init, crossover=SBX(prob=0.9, eta=5), mutation=PM(prob=mutation_prob, eta=5))

termination = get_termination("n_gen", n_gen)
start_time = time.time()
result = minimize(
    problem,
    algorithm,
    termination=termination,
    seed=42,
    save_history=True,
    verbose=True,
    callback=GenerationFrontCallback(),
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# extract results
if result.X is None:
    print("No feasible solutions found.")
    X_pareto = None
    F_pareto = None
else:
    X_pareto = result.X
    F_pareto = result.F
    Reg_pareto = result.opt.get("Reg")
    print("# of found Pareto solutions:", X_pareto.shape[0])
    if reg_as_objective:
        results_path = RUN_OUTPUT_DIR / f"{algo_name}_results_t1_EF_param_reg_{RUN_ID}.npz"
        csv_path = RUN_OUTPUT_DIR / f"{algo_name}_results_t1_EF_param_reg_{RUN_ID}.csv"
    else:
        results_path = RUN_OUTPUT_DIR / f"{algo_name}_results_t1_EF_reg_constrained_{RUN_ID}.npz"
        csv_path = RUN_OUTPUT_DIR / f"{algo_name}_results_t1_EF_reg_constrained_{RUN_ID}.csv"
    np.savez_compressed(
        results_path,
        X_pareto=X_pareto,
        F_pareto=F_pareto,
        Reg_pareto=Reg_pareto,
        parameter_names=np.asarray(names),
        objective_names=np.asarray(objective_names),
    )
    write_population_csv(csv_path, X_pareto, F_pareto, CV=None, extra_columns={"Reg": Reg_pareto})
    print(f"Results saved at {results_path}.")
    print("Printing all results:")
    print(X_pareto)
    print(F_pareto)

run_config["reg_as_objective"] = reg_as_objective
run_config["reg_cap"] = REG_CAP
run_config["use_baseline_constraint"] = use_baseline_constraint
run_config["baseline"] = {"E0": E0, "F0": F0}
run_config["algorithm"] = algo_name
run_config["number_of_objectives"] = number_of_objectives
(RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
