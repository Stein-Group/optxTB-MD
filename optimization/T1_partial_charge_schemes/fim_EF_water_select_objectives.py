"""
Cluster-level FIM analysis of GFN2-xTB parameters against (E, F) targets, with
a selectable set of FIM objectives.

This is the same analysis as fim_EF_water.py, generalized so the residual
vector fed into the Jacobian/FIM can be restricted to a subset of {E, F} via
a single keyword:

  fim_objectives = ("E", "F")   # full FIM (default, matches fim_EF_water.py)
  fim_objectives = ("F",)       # forces-only FIM
  fim_objectives = ("E",)       # energy-only FIM

The diagnostic scalar loss (printed when calculate_L=True, and used in the
directional scans) always reports both components regardless of
fim_objectives -- the keyword only controls which residual blocks are
included in the Jacobian/FIM itself.

Only water (T1 tier, train split) is used. Base code (data loading,
splitting, calculator, loss) is identical to nsga3_EF_water.py / fim_EF_water.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from sklearn.metrics import r2_score, root_mean_squared_error

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import (
    calculate_dftbplus_energy_reduced,
    calculate_e_unbiased_f_md,
    calculate_re_f_md,
    ensure_param_file,
    get_total_charge,
    optimize_dftbplus_geometry_reduced,
    read_xyz,
)
from codes.gradients_qchem import calculate_re_f_md_dft
from codes.common import (
    _env_bool,
    generate_parameter_info,
    compute_fim_from_jacobian,
    eigh_sorted,
    fim_condition_info,
    summarize_eigenvectors,
    write_csv_rows,
    plot_spectrum,
)

np.set_printoptions(threshold=np.inf)


# ----------------------- runtime setup ----------------------- #

def init_dftb_env(omp_threads=1, blas_threads=1, amd_mkl=False, bind_openmp=False):
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


init_dftb_env(omp_threads=1, blas_threads=1, amd_mkl=False, bind_openmp=False)

DEFAULT_SCRATCH_ROOT = Path(os.environ.get("SCRATCH", "/tmp"))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_PATH / "optimization/T1_partial_charge_schemes/hpc_runs"
DEFAULT_MONOMER_XYZ_ROOT = PROJECT_ROOT_PATH / "structures/cluster_data/monomers"


def _default_run_id():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slurm_job_id:
        return f"fim_t1_ef_water_obj_{slurm_job_id}_{stamp}"
    return f"fim_t1_ef_water_obj_local_{stamp}"



def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Run T1 water cluster FIM analysis (selectable objectives) on HPC.")
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

# ----------------------- FIM scientific settings ----------------------- #

calculate_L = True
calculate_FIM = True
finite_difference_h = 1e-2
finite_difference_method = "central"
run_directional_scans = True
scan_amplitudes = np.linspace(-0.5, 0.5, 9)
scan_mode_ids = "auto"  # "auto" = stiff, middle, sloppiest; or list of ints

# Keyword for FIM objective choice: any non-empty subset of {"E", "F"}.
# Only these residual blocks are concatenated into the FIM Jacobian; the
# diagnostic scalar loss below always reports both components regardless.
fim_objectives = ("E", "F") # "E", "F"
_valid_objectives = {"E", "F"}
if not fim_objectives or not set(fim_objectives).issubset(_valid_objectives):
    raise ValueError(f"fim_objectives must be a non-empty subset of {_valid_objectives}; got {fim_objectives}")
fim_objectives = tuple(o for o in ("E", "F") if o in fim_objectives)  # canonical order

wE = 1.0
wF = 1.0
alpha = 2.0
eps = 1e-12
penalty_value = 1e6

print(f"calculate_L: {calculate_L}, calculate_FIM: {calculate_FIM}")
print(f"FIM objectives: {fim_objectives}")
print(f"Loss weights: wE={wE}, wF={wF}")
print(f"FD method: {finite_difference_method}, h={finite_difference_h}")


# ----------------------- parameters ----------------------- #


param_changes = {
    "kpair": {
        "H-H": 1.0,
        "O-O": 1.0,
        "O-H": 1.0,
    },
    "H": {
        "gam": 0.4057710000000000,
        "lgam": [1.0000000000000000],
        "gam3": 0.0800000000000000,
        "zeff": 1.1053880000000000,
        "arep": 2.2137169999999999,
    },
    "O": {
        "gam": 0.4518960000000000,
        "lgam": [1.0000000000000000, 1.1497020000000000],
        "gam3": -0.0517134000000000,
        "zeff": 5.7844150000000001,
        "arep": 2.1657120000000001,
    },
}

names, bounds, default_gfn2_vector, structure = generate_parameter_info(param_changes, fixed_components=None)
print("Parameter names:")
print(names)
print("Bounds:")
print(bounds)
print(f"Number of included parameters {len(names)}")

bounds_arr = np.asarray(bounds, dtype=float)
lower = bounds_arr[:, 0]
upper = bounds_arr[:, 1]
param_scale = 0.5 * (upper - lower)
param_scale[param_scale == 0.0] = 1.0
theta0 = np.asarray(default_gfn2_vector, dtype=float)


# ----------------------- dataset ----------------------- #

dataset_path = str(PROJECT_ROOT_PATH / "structures/cluster_data/dataset")

use_oo_target = True  # append the 12 T1 O-O hexamer-network-scaling reference structures (320-331) to the training set
OO_TARGET_DIR = dataset_path + '/oo_target_small'
OO_TARGET_DFT_DIR = OO_TARGET_DIR + '/dft_results_revpbe0'

coord_files = []
pattern = re.compile(r'^\d+_.*\.xyz$')
for filename in os.listdir(dataset_path):
    match = pattern.match(filename)
    if match:
        file_number = int(match.group().split('_')[0])
        coord_files.append((file_number, filename))
coord_files.sort(key=lambda x: x[0])

DFT_LEVEL = "revpbe0"  # revPBE0-D3(0)/aug-cc-pVTZ references only

_DFT_BASE = str(PROJECT_ROOT_PATH / "structures/cluster_data/dataset")
dft_path = _DFT_BASE + '/dft_results_revpbe0'
dft_label = "revPBE0-D3/aug-cc-pVTZ"
dft_monomer_path = str(PROJECT_ROOT_PATH / "structures/cluster_data/monomers/revpbe0")
print(f"DFT reference: {dft_label} ({dft_path})")

print('Loading DFT reference data...')
res_ref = []
forces_ref = []
molecular_dipoles_ref = []
for _, filename in coord_files:
    coord_file_path = os.path.join(dataset_path, filename)
    RE_ref, F_ref, mu_ref = calculate_re_f_md_dft(coord_file_path, path=dft_path, path_monomers=dft_monomer_path)
    res_ref.append(RE_ref)
    forces_ref.append(np.array(F_ref))
    molecular_dipoles_ref.append(mu_ref)
res_ref = np.array(res_ref)
N_atoms_ref = np.array([np.asarray(forces_ref[i]).shape[0] for i in range(len(res_ref))], dtype=float)
if np.any(N_atoms_ref == 0):
    raise ValueError("Encountered structure with zero atoms.")


# ----------------------- calculator ----------------------- #

def get_monomer_energies(param_changes):
    key = json.dumps(param_changes, sort_keys=True, separators=(",", ":"))
    if key in MONOMER_CACHE:
        return MONOMER_CACHE[key]
    e_na = calculate_dftbplus_energy_reduced(
        param_changes, DEFAULT_MONOMER_XYZ_ROOT / "na/na.xyz", MONOMER_RUN_DIR / "na", charge=1,
    )
    e_cl = calculate_dftbplus_energy_reduced(
        param_changes, DEFAULT_MONOMER_XYZ_ROOT / "cl/cl.xyz", MONOMER_RUN_DIR / "cl", charge=-1,
    )
    e_water = optimize_dftbplus_geometry_reduced(
        param_changes, DEFAULT_MONOMER_XYZ_ROOT / "water/water.xyz", MONOMER_RUN_DIR / "water", charge=0,
    )
    MONOMER_CACHE[key] = (e_na, e_cl, e_water)
    return MONOMER_CACHE[key]


def calculate_dict(param_changes, coord_files):
    param_path = ensure_param_file(param_changes, DIR=WORKDIR)
    monomer_energies = get_monomer_energies(param_changes)

    def job(i_file):
        i, (file_number, filename) = i_file
        coord_file_path = os.path.join(dataset_path, filename)
        total_charge = get_total_charge(coord_file_path)
        run_dir = DFTB_RUNS / f"{i:05d}_{Path(filename).stem}__q{total_charge:+d}"
        RE, F, mu = calculate_re_f_md(
            param_changes, coord_file_path, total_charge,
            run_dir=run_dir, monomer_energies=monomer_energies,
            monomer_xyz_root=DEFAULT_MONOMER_XYZ_ROOT, param_path=param_path,
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

    failed = [i for i in range(len(res)) if res[i] is None or forces[i] is None or molecular_dipoles[i] is None]
    if failed:
        raise RuntimeError(
            f"Some DFTB+ jobs failed: {len(failed)}/{len(res)}. "
            f"Example error: {next(iter(errors.values())) if errors else 'unknown'}"
        )
    return np.array(res), forces, molecular_dipoles


def calculate(vector, coord_files):
    param_changes = vector_to_param_changes(vector, structure)
    return calculate_dict(param_changes, coord_files)


# ----------------------- dataset split (identical to nsga3_EF_water.py) ----------------------- #

split_dataset = True

if split_dataset:
    print('Splitting the dataset...')
    def classify_tier_and_ion(file_number, counts):
        n_na = counts.get('Na', 0)
        n_cl = counts.get('Cl', 0)
        if 1 <= file_number <= 81:
            tier = 'T1'; ion_type = 'none'
        elif 82 <= file_number <= 180:
            tier = 'T2_Na'; ion_type = 'Na+'
        elif 181 <= file_number <= 289:
            tier = 'T2_Cl'; ion_type = 'Cl-'
        elif 290 <= file_number <= 307:
            tier = 'T3'; ion_type = 'NaCl'
        else:
            if n_na == 0 and n_cl == 0:
                tier = 'T1'; ion_type = 'none'
            elif n_na > 0 and n_cl == 0:
                tier = 'T2_Na'; ion_type = 'Na+'
            elif n_na == 0 and n_cl > 0:
                tier = 'T2_Cl'; ion_type = 'Cl-'
            else:
                tier = 'T3'; ion_type = 'NaCl'
        return tier, ion_type

    def parse_isomer_and_distortion(filename_base):
        base = filename_base
        if 'mode' in base:
            prefix, mode_part = base.split('mode', 1)
            isomer_id = prefix.rstrip('_')
            m = re.match(r'(\d+)_?(minus|plus)', mode_part)
            if m:
                mode_id = int(m.group(1)); sign = m.group(2)
            else:
                mode_id = None; sign = None
        else:
            isomer_id = base; mode_id = None; sign = 'eq'
        return isomer_id.lower(), mode_id, sign

    metadata = []
    for file_number, filename in coord_files:
        coord_file_path = os.path.join(dataset_path, filename)
        _, symbols, _ = read_xyz(coord_file_path)
        counts = Counter(symbols)
        n_atoms = len(symbols)
        n_O = counts.get('O', 0); n_H = counts.get('H', 0)
        n_Na = counts.get('Na', 0); n_Cl = counts.get('Cl', 0)
        n_waters = n_O
        total_charge = get_total_charge(coord_file_path)
        tier, ion_type = classify_tier_and_ion(file_number, counts)
        base_part = filename.split('_', 1)[1].rsplit('.', 1)[0]
        isomer_id, mode_id, sign = parse_isomer_and_distortion(base_part)
        group_id = (tier, ion_type, n_waters, isomer_id)
        metadata.append(dict(
            idx=file_number, filename=filename, tier=tier, ion_type=ion_type,
            group_id=group_id, n_atoms=n_atoms, n_waters=n_waters,
            n_O=n_O, n_H=n_H, n_Na=n_Na, n_Cl=n_Cl,
            total_charge=total_charge, isomer_id=isomer_id, mode_id=mode_id, sign=sign,
        ))

    meta_df = pd.DataFrame(metadata).set_index("idx").sort_index()

    def evaluate_f_term_single(vector, coordinate_idx, F_denominator, coord_files, refs, penalty_value=1e2, eps=1e-12):
        F_ref_list = refs[1][coordinate_idx]
        try:
            _, F_pred_list, _ = calculate(vector, [coord_files[coordinate_idx]])
        except Exception as e:
            print(f"Exception during model calculation: {e}")
            return penalty_value
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
        for element in enumerate(coord_files):
            i = int(element[1][0])
            j = int(element[0])
            per_structure_F[i] = evaluate_f_term_single(default_gfn2_vector, j, 1.0, coord_files=coord_files, refs=(res_ref, forces_ref))
        return per_structure_F

    per_structure_F = make_f_measure(coord_files=coord_files, res_ref=res_ref, forces_ref=forces_ref)
    meta_df["F_term"] = meta_df.index.map(lambda i: per_structure_F.get(i, np.nan))

    def split_train_test(
        meta_df,
        test_frac=0.2,
        eq_to_train_tiers=("T1", "T2_Na", "T2_Cl"),
        tiers_order=("T1", "T2_Na", "T2_Cl"),
        add_T3=True,
    ):
        meta = meta_df.copy()
        if "F_term" not in meta.columns:
            raise ValueError("meta_df must contain 'F_term' column.")
        meta["F_term_eff"] = meta["F_term"]
        if meta["F_term_eff"].isna().any():
            print("Warning: NaN values found in 'F_term' column; filling with median.")

        train_indices = []
        test_indices = []
        for tier in tiers_order:
            tier_mask = (meta["tier"] == tier)
            tier_idx = meta.index[tier_mask].to_numpy()
            if len(tier_idx) == 0:
                continue
            is_eq = (meta.loc[tier_idx, "sign"] == "eq")
            if tier in eq_to_train_tiers:
                eq_idx = tier_idx[is_eq.to_numpy()]
                candidate_idx = tier_idx[~is_eq.to_numpy()]
            else:
                eq_idx = np.array([], dtype=int)
                candidate_idx = tier_idx
            n_tier = len(tier_idx)
            n_test_target = max(1, min(int(round(test_frac * n_tier)), len(candidate_idx)))
            if len(candidate_idx) == 0 or n_test_target <= 0:
                train_indices.extend(list(tier_idx))
                continue
            cand_df = meta.loc[candidate_idx].sort_values("F_term_eff")
            cand_sorted = cand_df.index.to_numpy()
            if n_test_target >= len(cand_sorted):
                chosen_test = cand_sorted
            else:
                positions = np.unique(np.linspace(0, len(cand_sorted) - 1, num=n_test_target, dtype=int))
                chosen_test = cand_sorted[positions]
            test_indices.extend(list(chosen_test))
            train_indices.extend(list(np.setdiff1d(tier_idx, chosen_test)))

        train_indices = np.array(sorted(set(train_indices)))
        test_indices = np.array(sorted(set(test_indices)))
        inter = np.intersect1d(train_indices, test_indices)
        if len(inter) > 0:
            raise RuntimeError(f"Train/Test overlap in indices: {inter}")
        if add_T3:
            train_indices = np.concatenate([train_indices, [290, 291, 293, 294]])
            test_indices = np.concatenate([test_indices, [292, 295]])
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
    train_coord = coord_files[:81]
    train_refs = ([res_ref[i] for i in range(81)], [forces_ref[i] for i in range(81)])
    train_N_atoms = N_atoms_ref[:81]
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
        RE, F, _ = calculate_re_f_md_dft(coord_file_path, path=OO_TARGET_DFT_DIR, path_monomers=dft_monomer_path)
        oo_res.append(RE)
        oo_forces.append(np.array(F))
    train_coord = train_coord + [(n, f"oo_target_small/{f}") for n, f in oo_target_files]
    train_refs = (np.concatenate([train_refs[0], np.array(oo_res)]), list(train_refs[1]) + oo_forces)
    train_N_atoms = np.concatenate([train_N_atoms, np.array([len(f) for f in oo_forces], dtype=float)])

print(f'Number of training structures: {len(train_coord)}')


# ----------------------- run config ----------------------- #

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
    "calculate_L": calculate_L,
    "calculate_FIM": calculate_FIM,
    "finite_difference_h": finite_difference_h,
    "finite_difference_method": finite_difference_method,
    "fim_objectives": list(fim_objectives),
    "weights": {"E": wE, "F": wF},
    "fim_parameter_names": names,
    "fim_bounds": bounds,
    "fim_param_scale": param_scale.tolist(),
}
(RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
print(f"Run config saved to {RUN_OUTPUT_DIR / 'run_config.json'}")


# ----------------------- loss (always full E/F diagnostic) ----------------------- #

def param_regularization(vector, theta0=theta0, scale=param_scale):
    v = np.asarray(vector, dtype=float)
    diff = (v - theta0) / scale
    return float(np.mean(diff * diff))


def loss(vector, coord_files=train_coord, wE=wE, wF=wF, refs=train_refs, N_atoms=train_N_atoms, penalty_value=penalty_value, eps=eps, return_components=False, print_components=False, force_term_type="mse", alpha=alpha):
    E_ref = np.asarray(refs[0], dtype=float)
    F_ref_list = refs[1]
    try:
        E_pred, F_pred_list, _ = calculate(vector, coord_files)
    except Exception as e:
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})

    if E_pred is None or F_pred_list is None:
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
    n = len(E_ref)
    if not (len(E_pred) == len(F_ref_list) == len(F_pred_list) == n):
        return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})

    dE = np.asarray(E_pred, dtype=float) - E_ref
    wE_i = 1 / N_atoms
    sum_wE = np.sum(wE_i)
    mean_dE = np.sum(wE_i * dE) / sum_wE
    mean_Eref = np.sum(wE_i * E_ref) / sum_wE
    E_numerator = np.sum(wE_i * (dE - mean_dE) ** 2)
    E_denominator = np.sum(wE_i * (E_ref - mean_Eref) ** 2)

    F_numerator = 0.0
    F_denominator = 0.0
    for i in range(n):
        Fref = np.asarray(F_ref_list[i], dtype=float)
        Fpred = np.asarray(F_pred_list[i], dtype=float)
        if Fref.shape != Fpred.shape or Fref.ndim != 2 or Fref.shape[1] != 3:
            return penalty_value if not return_components else (penalty_value, {"E": penalty_value, "F": penalty_value, "Reg": penalty_value})
        Ni = Fref.shape[0]
        dF = Fpred - Fref
        wF_i = 1 / (3 * Ni)
        F_numerator += wF_i * float(np.sum(dF * dF))
        F_denominator += wF_i * float(np.sum(Fref * Fref))

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


force_term_type = "mse"
print(f"Selected force term type: {force_term_type}")


# ----------------------- initial parity plots ----------------------- #

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

    F_ref_all = np.concatenate([F_ref_list_conv[i] for i in range(len(F_ref_list_conv))], axis=0).ravel()
    F_pred_all = np.concatenate([F_pred_list_conv[i] for i in range(len(F_pred_list_conv))], axis=0).ravel()
    print(f"[{set_name}] Force RMSE (GFN2 default): {root_mean_squared_error(F_ref_all, F_pred_all):.6f} meV/Å")

    component_names = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=False, sharey=False)
    for comp, ax in enumerate(axes):
        F_ref = np.concatenate([F_ref_list_conv[i][:, comp] for i in range(len(F_ref_list_conv))])
        F_pred = np.concatenate([F_pred_list_conv[i][:, comp] for i in range(len(F_pred_list_conv))])
        ax.scatter(F_ref, F_pred, alpha=0.5, marker="o",
                   label="GFN2-xTB, R$^2$={:.4f}, RMSE={:.2f} meV/Å".format(r2_score(F_ref, F_pred), root_mean_squared_error(F_ref, F_pred)))
        vmin = min(F_ref.min(), F_pred.min()); vmax = max(F_ref.max(), F_pred.max())
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
    plt.savefig(out_forces, dpi=150); plt.close()
    print(f"Saved {out_forces}")

    scalar_product = []; magnitude_pred = []
    for struc in range(len(coord_files)):
        F_ref = np.asarray(F_ref_list_conv[struc], dtype=float)
        F_pred = np.asarray(F_pred_list_conv[struc], dtype=float)
        for atom in range(len(F_ref)):
            mag_pred = np.linalg.norm(F_pred[atom])
            unit_ref = F_ref[atom] / (np.linalg.norm(F_ref[atom]) + 1e-12)
            unit_pred = F_pred[atom] / (mag_pred + 1e-12)
            scalar_product.append(1 - np.dot(unit_ref, unit_pred))
            magnitude_pred.append(mag_pred)
    scalar_product = np.array(scalar_product); magnitude_pred = np.array(magnitude_pred)
    print(f"[{set_name}] GFN2 directional deviation: mean={np.mean(scalar_product * magnitude_pred):.6f}, median={np.median(scalar_product * magnitude_pred):.6f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.arange(len(scalar_product)), scalar_product * magnitude_pred, alpha=0.5, color='tab:blue',
               label=f'Mean {np.mean(scalar_product * magnitude_pred):.3f}, median {np.median(scalar_product * magnitude_pred):.3f} meV/Å')
    ax.set_title(f'GFN2: Directional Deviation ({set_name} set)')
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('[1 - (scalar product of force unit vectors)]*(force magnitude per atom)', fontsize=9)
    ax.grid(); ax.legend()
    plt.tight_layout()
    out_dir = RUN_OUTPUT_DIR / f"initial_parity_{set_name}_directions.png"
    plt.savefig(out_dir, dpi=150); plt.close()
    print(f"Saved {out_dir}")

    E_pred_conv = np.array(E_pred_list) * hartree_to_kcalmol
    E_ref_conv = np.array(refs[0]) * hartree_to_kcalmol
    print(f"[{set_name}] Energy RMSE (GFN2 default): {root_mean_squared_error(E_ref_conv, E_pred_conv):.3f} kcal/mol")
    plt.figure(figsize=(5, 5))
    plt.scatter(E_ref_conv, E_pred_conv, alpha=0.5, marker='o',
                label='GFN2-xTB, R^2 = {:.4f}, \nRMSE = {:.3f} kcal/mol'.format(r2_score(E_ref_conv, E_pred_conv), root_mean_squared_error(E_ref_conv, E_pred_conv)))
    plt.xlabel(f"Reference Energies ({dft_label}) (kcal/mol)")
    plt.ylabel("Predicted Energies (kcal/mol)")
    plt.title(f"Scatter plot of Energies ({set_name} set) - GFN2 default")
    plt.plot([E_ref_conv.min(), E_ref_conv.max()], [E_ref_conv.min(), E_ref_conv.max()], 'r--', label='y=x')
    plt.grid(); plt.legend()
    out_ene = RUN_OUTPUT_DIR / f"initial_parity_{set_name}_energies.png"
    plt.savefig(out_ene, dpi=150); plt.close()
    print(f"Saved {out_ene}")


# ----------------------- FIM machinery (objective-selectable) ----------------------- #

@dataclass
class ResidualNormCache:
    denominators: dict = field(default_factory=dict)  # objective -> normalization denominator
    wE_i: np.ndarray = None
    sizes: dict = field(default_factory=dict)  # objective -> residual block size


def make_residual_norm_cache(refs, N_atoms, fim_objectives=fim_objectives, eps=eps):
    """Compute normalisation denominators from pre-loaded DFT references (cluster-specific)."""
    E_ref = np.asarray(refs[0], dtype=float)
    F_ref_list = refs[1]
    wE_i = 1.0 / np.asarray(N_atoms, dtype=float)

    denominators = {}
    sizes = {}

    if "E" in fim_objectives:
        mean_Eref = np.sum(wE_i * E_ref) / np.sum(wE_i)
        denominators["E"] = float(np.sum(wE_i * (E_ref - mean_Eref) ** 2))
        sizes["E"] = len(E_ref)

    if "F" in fim_objectives:
        F_denominator = 0.0
        nF = 0
        for F in F_ref_list:
            Fref = np.asarray(F, dtype=float)
            Ni = Fref.shape[0]
            F_denominator += (1.0 / (3.0 * Ni)) * float(np.sum(Fref * Fref))
            nF += Fref.size
        denominators["F"] = float(F_denominator)
        sizes["F"] = int(nF)

    for obj, denom in denominators.items():
        if not np.isfinite(denom) or denom < eps:
            raise ValueError(f"Bad {obj}_denominator: {denom}")

    return ResidualNormCache(denominators=denominators, wE_i=wE_i, sizes=sizes)


_calculate_cache: dict = {}


def calculate_cached(vector, coord_files):
    key = (tuple(np.round(np.asarray(vector, dtype=float), 14)), tuple(fn for _, fn in coord_files))
    if key not in _calculate_cache:
        _calculate_cache[key] = calculate(vector, coord_files)
    return _calculate_cache[key]


def residual_vector(vector, coord_files, refs, norm_cache, wE=wE, wF=wF, eps=eps, return_slices=False, fim_objectives=fim_objectives):
    """Build r such that dot(r, r) reproduces the scalar cluster loss restricted to
    the objectives in fim_objectives (a subset of {E, F})."""
    weights = {"E": wE, "F": wF}
    E_pred, F_pred_list, _ = calculate_cached(vector, coord_files)
    E_ref = np.asarray(refs[0], dtype=float)

    parts = {}

    if "E" in fim_objectives:
        dE = np.asarray(E_pred, dtype=float) - E_ref
        mean_dE = np.sum(norm_cache.wE_i * dE) / np.sum(norm_cache.wE_i)
        parts["E"] = np.sqrt(weights["E"] * norm_cache.wE_i / (norm_cache.denominators["E"] + eps)) * (dE - mean_dE)

    if "F" in fim_objectives:
        rF_parts = []
        for i, F_pred in enumerate(F_pred_list):
            Fref = np.asarray(refs[1][i], dtype=float)
            Fpred = np.asarray(F_pred, dtype=float)
            Ni = Fref.shape[0]
            rF_parts.append(
                (np.sqrt(weights["F"] / (3.0 * Ni) / (norm_cache.denominators["F"] + eps)) * (Fpred - Fref)).reshape(-1)
            )
        parts["F"] = np.concatenate(rF_parts)

    r = np.concatenate([parts[obj] for obj in fim_objectives])

    if not np.all(np.isfinite(r)):
        raise RuntimeError("Non-finite residual vector encountered.")

    if return_slices:
        slices = {}
        i0 = 0
        for obj in fim_objectives:
            slices[obj] = slice(i0, i0 + parts[obj].size)
            i0 = slices[obj].stop
        return r, slices, parts
    return r


def compute_jacobian_fd_scaled(theta_ref, residual_func, scale, lower, upper, h=1e-2, method="central", verbose=True):
    theta_ref = np.asarray(theta_ref, dtype=float).copy()
    scale = np.asarray(scale, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("All scale entries must be > 0.")
    r0 = residual_func(theta_ref)
    J = np.zeros((r0.size, theta_ref.size), dtype=float)
    steps = np.zeros(theta_ref.size, dtype=float)
    fd_mode = []

    for j in range(theta_ref.size):
        max_plus_scaled = (upper[j] - theta_ref[j]) / scale[j]
        max_minus_scaled = (theta_ref[j] - lower[j]) / scale[j]
        if verbose:
            print(f"[{j + 1:02d}/{theta_ref.size:02d}] {names[j]}")
        ej = np.zeros(theta_ref.size, dtype=float)
        ej[j] = 1.0

        if method == "central":
            h_j = min(h, 0.49 * max_plus_scaled, 0.49 * max_minus_scaled)
            if np.isfinite(h_j) and h_j > 0:
                r_p = residual_func(theta_ref + h_j * scale * ej)
                r_m = residual_func(theta_ref - h_j * scale * ej)
                J[:, j] = (r_p - r_m) / (2.0 * h_j)
                steps[j] = h_j
                fd_mode.append("central")
                continue

        h_plus = min(h, 0.49 * max_plus_scaled)
        h_minus = min(h, 0.49 * max_minus_scaled)
        if np.isfinite(h_plus) and h_plus > 0:
            r_p = residual_func(theta_ref + h_plus * scale * ej)
            J[:, j] = (r_p - r0) / h_plus
            steps[j] = h_plus
            fd_mode.append("forward")
        elif np.isfinite(h_minus) and h_minus > 0:
            r_m = residual_func(theta_ref - h_minus * scale * ej)
            J[:, j] = (r0 - r_m) / h_minus
            steps[j] = h_minus
            fd_mode.append("backward")
        else:
            raise RuntimeError(f"Could not find a valid FD step for parameter {j}: {names[j]}")

    return J, r0, steps, fd_mode







def save_eigenvalue_csv(evals_by_block):
    rows = []
    for block, evals in evals_by_block.items():
        total = float(np.sum(np.clip(evals, 0.0, None)))
        for mode, lam in enumerate(evals):
            rows.append({
                "block": block, "mode": mode, "eigenvalue": float(lam),
                "relative_to_max": float(lam / evals[0]) if evals[0] != 0 else np.nan,
                "cumulative_trace_fraction": float(np.sum(np.clip(evals[:mode + 1], 0.0, None)) / total) if total > 0 else np.nan,
            })
    write_csv_rows(RUN_OUTPUT_DIR / "fim_eigenvalues.csv", rows,
                   ["block", "mode", "eigenvalue", "relative_to_max", "cumulative_trace_fraction"])


def save_eigenvector_csv(evals, evecs):
    rows = []
    for mode, lam in enumerate(evals):
        for pidx, name in enumerate(names):
            coeff = float(evecs[pidx, mode])
            rows.append({"mode": mode, "eigenvalue": float(lam), "parameter": name,
                         "coefficient": coeff, "participation": coeff * coeff})
    write_csv_rows(RUN_OUTPUT_DIR / "fim_total_eigenvectors.csv", rows,
                   ["mode", "eigenvalue", "parameter", "coefficient", "participation"])



def plot_mode_participation_heatmap(evecs, mode_ids, out_path, title="Participation factor for all modes"):
    V = np.asarray(evecs[:, mode_ids], dtype=float)
    P = V ** 2
    P_masked = np.ma.masked_where(P <= 0.0, P)
    fig_width = max(8.0, 0.35 * len(mode_ids) + 4.0)
    fig_height = max(5.0, 0.35 * len(names) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.cm.viridis.copy(); cmap.set_bad(color="white")
    im = ax.imshow(P_masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(mode_ids)))
    ax.set_xticklabels([f"mode_{m}" for m in mode_ids], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names)
    ax.set_title(title); fig.colorbar(im, ax=ax, label="v_jk^2")
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def block_contributions_for_total_modes(fim_blocks, evals_total, evecs_total, fim_objectives=fim_objectives):
    rows = []
    for mode in range(len(evals_total)):
        lam = float(evals_total[mode])
        v = evecs_total[:, mode]
        denom = lam if lam > 0 else np.nan
        row = {"mode": mode, "lambda_total": lam}
        for obj in fim_objectives:
            c = float(v.T @ fim_blocks[obj] @ v)
            row[f"{obj}_abs"] = c
            row[f"{obj}_frac"] = c / denom
        rows.append(row)
    return rows


def plot_block_fractions(rows, out_path, fim_objectives=fim_objectives, title="Block contribution fractions for all modes"):
    x = np.arange(len(rows))
    modes = [row["mode"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(9.0, 0.35 * len(rows) + 4.0), 4.5))
    bottom = np.zeros(len(rows))
    for obj in fim_objectives:
        frac = np.asarray([row[f"{obj}_frac"] for row in rows], dtype=float)
        ax.bar(x, frac, bottom=bottom, label=obj)
        bottom = bottom + frac
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_xlabel("Total mode"); ax.set_ylabel("Fraction of lambda_total")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def scan_along_eigendirection(theta_ref, vec_scaled, amplitudes, scale, lower, upper):
    rows = []
    for amp in amplitudes:
        theta = np.asarray(theta_ref, dtype=float) + amp * (scale * vec_scaled)
        hit_lower = bool(np.any(theta < lower))
        hit_upper = bool(np.any(theta > upper))
        theta = np.minimum(np.maximum(theta, lower), upper)
        L, comps = loss(theta, return_components=True, print_components=False)
        rows.append({"amp": float(amp), "L": float(L), "E": float(comps["E"]),
                     "F": float(comps["F"]), "hit_lower": hit_lower, "hit_upper": hit_upper})
    return rows


def plot_directional_scan(rows, lambda_k, out_path, title):
    amp = np.asarray([row["amp"] for row in rows], dtype=float)
    L = np.asarray([row["L"] for row in rows], dtype=float)
    E = np.asarray([row["E"] for row in rows], dtype=float)
    F = np.asarray([row["F"] for row in rows], dtype=float)
    zero_idx = int(np.argmin(np.abs(amp)))
    L_quad = L[zero_idx] + lambda_k * amp ** 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(amp, L, marker="o", label="actual total loss")
    axes[0].plot(amp, L_quad, linestyle="--", label="quadratic L0 + lambda_k a^2")
    axes[0].set_xlabel("Amplitude along eigendirection"); axes[0].set_ylabel("Loss")
    axes[0].set_title(title); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(amp, E, marker="o", label="E")
    axes[1].plot(amp, F, marker="o", label="F")
    axes[1].set_xlabel("Amplitude"); axes[1].set_ylabel("Component value")
    axes[1].set_title(title + " components"); axes[1].legend(); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def run_fim_analysis():
    theta_ref = np.asarray(default_gfn2_vector, dtype=float)
    scale = np.asarray(param_scale, dtype=float)

    norm_cache = make_residual_norm_cache(train_refs, train_N_atoms, fim_objectives=fim_objectives)
    r_ref, slices_ref, parts_ref = residual_vector(
        theta_ref, train_coord, train_refs, norm_cache, return_slices=True, fim_objectives=fim_objectives,
    )
    L0_from_r = float(np.dot(r_ref, r_ref))
    weighted_residual_block_terms = {key: float(np.dot(v, v)) for key, v in parts_ref.items()}
    print("Residual sizes:")
    for obj in fim_objectives:
        print(f"  {obj}:", parts_ref[obj].size)
    print("  total:", r_ref.size)
    print("||r||^2 =", L0_from_r)
    print("Weighted residual block contributions:", weighted_residual_block_terms)

    def residual_func(theta_raw):
        return residual_vector(theta_raw, train_coord, train_refs, norm_cache, return_slices=False, fim_objectives=fim_objectives)

    start_time = time.time()
    Jz, r0, steps_used, fd_mode = compute_jacobian_fd_scaled(
        theta_ref=theta_ref, residual_func=residual_func, scale=scale, lower=lower, upper=upper,
        h=finite_difference_h, method=finite_difference_method, verbose=True,
    )
    fim_time = time.time() - start_time

    fim_blocks = compute_fim_from_jacobian(Jz, slices=slices_ref)
    evals_total, evecs_total = eigh_sorted(fim_blocks["total"])
    evals_by_obj = {}
    evecs_by_obj = {}
    for obj in fim_objectives:
        evals_by_obj[obj], evecs_by_obj[obj] = eigh_sorted(fim_blocks[obj])

    print("\nSteps actually used:")
    for name_p, h_j, mode in zip(names, steps_used, fd_mode):
        print(f"{name_p:20s}  h={h_j:.5e}  mode={mode}")

    print("\nFIM condition info (total):")
    print(fim_condition_info(evals_total))
    loading_rows = []
    loading_rows.extend(summarize_eigenvectors(evals_total, evecs_total, names, n_show=10, top_loadings=8, which="stiff"))
    loading_rows.extend(summarize_eigenvectors(evals_total, evecs_total, names, n_show=10, top_loadings=8, which="sloppy"))
    for obj in fim_objectives:
        print(f"\nFIM condition info ({obj} block):")
        print(fim_condition_info(evals_by_obj[obj]))
        loading_rows.extend(summarize_eigenvectors(evals_by_obj[obj], evecs_by_obj[obj], names, n_show=5, top_loadings=6, which="stiff"))
        loading_rows.extend(summarize_eigenvectors(evals_by_obj[obj], evecs_by_obj[obj], names, n_show=5, top_loadings=6, which="sloppy"))

    np.savez_compressed(
        RUN_OUTPUT_DIR / "fim_results_default_gfn2.npz",
        Jz=Jz, r0=r0, steps_used=steps_used, fd_mode=np.asarray(fd_mode),
        evals_total=evals_total, evecs_total=evecs_total,
        F_total=fim_blocks["total"],
        names=np.asarray(names), theta_ref=theta_ref, scale=scale, lower=lower, upper=upper,
        fim_objectives=np.asarray(fim_objectives),
        **{f"evals_{obj}": evals_by_obj[obj] for obj in fim_objectives},
        **{f"evecs_{obj}": evecs_by_obj[obj] for obj in fim_objectives},
        **{f"F_{obj}": fim_blocks[obj] for obj in fim_objectives},
    )

    save_eigenvalue_csv({"total": evals_total, **evals_by_obj})
    save_eigenvector_csv(evals_total, evecs_total)
    write_csv_rows(RUN_OUTPUT_DIR / "fim_top_loadings.csv", loading_rows,
                   ["which", "mode", "eigenvalue", "rank", "parameter", "coefficient", "participation"])

    plot_spectrum(evals_total, "total", RUN_OUTPUT_DIR / "fim_spectrum_total.png")
    spectrum_label = {"E": "energy", "F": "forces"}
    for obj in fim_objectives:
        plot_spectrum(evals_by_obj[obj], spectrum_label[obj], RUN_OUTPUT_DIR / f"fim_spectrum_{spectrum_label[obj]}.png")

    mode_ids = list(range(len(evals_total)))[::-1]
    plot_mode_participation_heatmap(
        evecs_total, mode_ids, RUN_OUTPUT_DIR / "fim_participation_heatmap_all_modes.png",
        title="Participation factor for all modes (sloppy to stiff)",
    )

    block_rows = block_contributions_for_total_modes(fim_blocks, evals_total, evecs_total, fim_objectives=fim_objectives)
    block_fields = ["mode", "lambda_total"]
    for obj in fim_objectives:
        block_fields.extend([f"{obj}_abs", f"{obj}_frac"])
    write_csv_rows(RUN_OUTPUT_DIR / "fim_block_contribution_fractions.csv", block_rows, block_fields)
    plot_block_fractions(block_rows, RUN_OUTPUT_DIR / "fim_block_contribution_fractions_all_modes.png", fim_objectives=fim_objectives)

    scan_rows_all = []
    if run_directional_scans:
        if scan_mode_ids == "auto":
            selected_modes = sorted(set([0, len(evals_total) // 2, len(evals_total) - 1]))
        else:
            selected_modes = list(scan_mode_ids)
        for mode in selected_modes:
            rows = scan_along_eigendirection(theta_ref, evecs_total[:, mode], scan_amplitudes, scale, lower, upper)
            for row in rows:
                row["mode"] = int(mode); row["lambda_total"] = float(evals_total[mode])
            scan_rows_all.extend(rows)
            plot_directional_scan(rows, float(evals_total[mode]),
                                  RUN_OUTPUT_DIR / f"fim_directional_scan_mode_{mode:03d}.png",
                                  title=f"Mode {mode} scan")
        write_csv_rows(RUN_OUTPUT_DIR / "fim_directional_scans.csv", scan_rows_all,
                       ["mode", "lambda_total", "amp", "L", "E", "F", "hit_lower", "hit_upper"])

    summary = {
        "run_id": RUN_ID,
        "n_train_structures": len(train_coord),
        "n_parameters": len(names),
        "parameter_names": names,
        "dft_label": dft_label,
        "fim_objectives": list(fim_objectives),
        "theta_ref_label": "default_gfn2",
        "finite_difference_h": finite_difference_h,
        "finite_difference_method": finite_difference_method,
        "steps_used": steps_used.tolist(),
        "fd_mode": fd_mode,
        "residual_sizes": {key: int(v.size) for key, v in parts_ref.items()},
        "residual_size_total": int(r_ref.size),
        "loss_from_residual_dot": L0_from_r,
        "weighted_loss_blocks_from_residual_dot": weighted_residual_block_terms,
        "fim_wall_time_seconds": float(fim_time),
        "condition_info": {
            "total": fim_condition_info(evals_total),
            **{obj: fim_condition_info(evals_by_obj[obj]) for obj in fim_objectives},
        },
    }
    (RUN_OUTPUT_DIR / "fim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ----------------------- run blocks ----------------------- #

if calculate_L:
    print("Running reference loss calculation...")
    start_time = time.time()
    L = loss(default_gfn2_vector, print_components=True, force_term_type=force_term_type)
    sweep_time = time.time() - start_time
    print(f"Sweep time: {sweep_time:.2f} seconds")
    print(f"Loss: {L}")
    print("Generating initial parity plots...")
    plot_initial_parity(train_coord, train_refs, "train")
    if test_coord is not None:
        plot_initial_parity(test_coord, test_refs, "test")

if calculate_FIM:
    print("Running cluster water FIM analysis...")
    run_fim_analysis()
