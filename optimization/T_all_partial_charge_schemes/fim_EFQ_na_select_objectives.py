"""
Cluster NaCl(aq) FIM analysis for Na-only parameters around the default GFN2-xTB
point, with a selectable set of FIM objectives.

This is the same analysis as fim_EFQ_na.py, generalized so the residual vector
fed into the Jacobian/FIM can be restricted to a subset of {E, F, Q} via a
single keyword:

  fim_objectives = ("E", "F", "Q")   # full FIM (default, matches fim_EFQ_na.py)
  fim_objectives = ("E", "F")        # drop partial charges from the FIM
  fim_objectives = ("Q",)            # charges-only FIM
  ...

The diagnostic scalar loss (printed when calculate_L=True, and used in the
directional scans) always reports all three components (E, F, Q) regardless
of fim_objectives -- the keyword only controls which residual blocks are
included in the Jacobian/FIM itself.

H/O element parameters (zeff, arep) and O-O/O-H kpair terms are fixed to the
optimised cluster water-only values loaded from FIXED_HO_TOML_PATH (same
narrow scope as pbc_fim_Na.py). Only Na element parameters (s-orbital levels,
slater, gam, lgam[1], gam3, zeff, arep) plus Na-O/Na-H kpair terms are varied.

Usage:

  python -u optimization/T_all_partial_charge_schemes/fim_EFQ_na_select_objectives.py --run-id fim_efq_na_obj_smoke

Submit on the cluster with:

  sbatch optimization/T_all_partial_charge_schemes/submit_fim_EFQ_na_select_objectives.slurm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import toml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import get_total_charge, read_xyz
from codes.molden_handling import MoldenTemplateCache, prepare_molden_templates_gfn2
from codes.tblite_gradients import calculate_dict_tblite_multiwfn_parallel_charges
from codes.multiwfn_qchem import load_dft_reference_dataset_with_scheme, precompute_multiwfn_charges_from_fchk
from codes.common import (
    _env_bool,
    generate_parameter_info,
    load_fixed_params_from_toml,
    merge_param_changes,
    is_na_tier,
    _vector_cache_key,
    compute_fim_from_jacobian,
    eigh_sorted,
    fim_condition_info,
    summarize_eigenvectors,
    write_csv_rows,
    plot_spectrum,
    parity_plot,
)


# ----------------------- runtime setup ----------------------- #

def _default_scratch_root():
    candidates = [Path(p) for p in (os.environ.get("SCRATCH"), "/tmp") if p]
    for path in candidates:
        if path.exists() and os.access(path, os.W_OK):
            return path
    return Path("/tmp")


DEFAULT_SCRATCH_ROOT = _default_scratch_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_PATH / "optimization/T_all_partial_charge_schemes/hpc_runs"
DEFAULT_MULTIWFN_EXE = "Multiwfn"
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/molden_templates_gfn2"
DEFAULT_ATMRAD_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/multiwfn_atmrad"

FIXED_HO_TOML_PATH = str(PROJECT_ROOT_PATH / "parameter_files/nsga3_EF_water_full_reg_6_params_249.toml")
FIXED_ELEMENT_PARAMS = {
    "H": ("zeff", "arep"),
    "O": ("zeff", "arep"),
}
FIXED_KPAIR_KEYS = ("O-O","O-H",)


def _default_run_id():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slurm_job_id:
        return f"fim_t2na_t3_efq_na_obj_{slurm_job_id}_{stamp}"
    return f"fim_t2na_t3_efq_na_obj_local_{stamp}"



def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Run cluster Na-only FIM analysis (selectable objectives) safely on HPC.")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", _default_run_id()))
    parser.add_argument("--scratch-root", default=os.environ.get("SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT)))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--multiwfn-threads", type=int, default=None)
    parser.add_argument("--multiwfn-exe", default=os.environ.get("MULTIWFN_EXE", DEFAULT_MULTIWFN_EXE))
    parser.add_argument("--template-dir", default=os.environ.get("TEMPLATE_DIR", str(DEFAULT_TEMPLATE_DIR)))
    parser.add_argument("--atmrad-dir", default=os.environ.get("ATMRAD_DIR", str(DEFAULT_ATMRAD_DIR)))
    parser.add_argument("--prepare-templates", action="store_true", default=_env_bool("PREPARE_TEMPLATES", False))
    parser.add_argument("--allow-oversubscription", action="store_true", default=_env_bool("ALLOW_OVERSUBSCRIPTION", False))
    args, _unknown = parser.parse_known_args()

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    args.multiwfn_threads = args.multiwfn_threads or int(os.environ.get("MULTIWFN_THREADS", 4))
    env_workers = os.environ.get("N_WORKERS")
    requested_workers = args.n_workers if args.n_workers is not None else (int(env_workers) if env_workers else None)
    args.n_workers = requested_workers or max(1, slurm_cpus // args.multiwfn_threads)
    if args.n_workers * args.multiwfn_threads > slurm_cpus and not args.allow_oversubscription:
        reduced_workers = max(1, slurm_cpus // args.multiwfn_threads)
        print(
            f"Warning: N_WORKERS={args.n_workers} with MULTIWFN_THREADS={args.multiwfn_threads} "
            f"exceeds SLURM_CPUS_PER_TASK={slurm_cpus}. Reducing N_WORKERS to {reduced_workers}."
        )
        args.n_workers = reduced_workers
    args.slurm_cpus = slurm_cpus
    return args


HPC = parse_hpc_args()
RUN_ID = HPC.run_id
SCRATCH_DIR = Path(HPC.scratch_root) / "optxTB-MD" / RUN_ID
WORKDIR = SCRATCH_DIR / "work"
RUN_OUTPUT_DIR = Path(HPC.output_root) / RUN_ID
TBLITE_RUNS = WORKDIR / "tblite_runs"
MOLDEN_OUT = WORKDIR / "molden_tblite_current"
MULTIWFN_ROOT = WORKDIR / "multiwfn_runs"
MONOMER_RUN_DIR = WORKDIR / "monomers_dftb_runs"
MULTIWFN_REF_PRECOMPUTE_DIR = WORKDIR / "multiwfn_ref_precompute"
TEMPLATE_DIR = Path(HPC.template_dir)
ATMRAD_DIR = Path(HPC.atmrad_dir)
MULTIWFN_EXE = HPC.multiwfn_exe
MULTIWFN_THREADS = HPC.multiwfn_threads
N_WORKERS = HPC.n_workers
PREPARE_TEMPLATES = HPC.prepare_templates

WORKDIR.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

print(f"RUN_ID: {RUN_ID}")
print(f"Scratch workdir: {WORKDIR}")
print(f"Persistent output dir: {RUN_OUTPUT_DIR}")
print(f"Multiwfn executable: {MULTIWFN_EXE}")
print(f"Multiwfn threads per worker: {MULTIWFN_THREADS}")
print(f"Python workers: {N_WORKERS}")
print(f"Molden template dir: {TEMPLATE_DIR}")
print(f"Multiwfn atmrad dir: {ATMRAD_DIR}")


# ----------------------- scientific settings ----------------------- #

dataset_path = PROJECT_ROOT_PATH / "structures/cluster_data/dataset"
dft_out_dir = dataset_path / "dft_results_revpbe0"
monomer_out_dir = PROJECT_ROOT_PATH / "structures/cluster_data/monomers/revpbe0"
chg_root = dataset_path / "dft_multiwfn_chg_revpbe0"
# revPBE0 .fchk files are only read if a reference charge (.chg) is missing from chg_root
FCHK_ROOT = Path(os.environ.get("FCHK_ROOT", str(PROJECT_ROOT_PATH / "structures/cluster_data/fchk")))
fchk_dir = FCHK_ROOT / "dft_results_revpbe0_fchk"

use_na_o_target = False  # append the 6 Na-O 1:1 rigid-scan reference structures (308-313) to the FIM evaluation set
NA_O_TARGET_DIR = dataset_path / "na_o_target"
NA_O_TARGET_DFT_DIR = NA_O_TARGET_DIR / "dft_results_revpbe0"
NA_O_TARGET_FCHK_DIR = FCHK_ROOT / "dft_results_revpbe0_fchk_na_o_target"
NA_O_TARGET_CHG_DIR = NA_O_TARGET_DIR / "dft_multiwfn_chg_revpbe0"

use_na_h2o4_target = False  # append the 6 Na(H2O)4 tetrahedral rigid-scan reference structures (314-319) to the FIM evaluation set
NA_H2O4_TARGET_DIR = dataset_path / "na_h2o4_target"
NA_H2O4_TARGET_DFT_DIR = NA_H2O4_TARGET_DIR / "dft_results_revpbe0"
NA_H2O4_TARGET_FCHK_DIR = FCHK_ROOT / "dft_results_revpbe0_fchk_na_h2o4_target"
NA_H2O4_TARGET_CHG_DIR = NA_H2O4_TARGET_DIR / "dft_multiwfn_chg_revpbe0"

type_charge = "hirshfeldI"  # "CM5" or "hirshfeldI"
print(f"Selected partial charge scheme: {type_charge}")

MAX_STRUCTURES_FOR_TEST = None  # set to a small int for a quick smoke test

# Keyword for FIM objective choice: any non-empty subset of {"E", "F", "Q"}.
# Only these residual blocks are concatenated into the FIM Jacobian; the
# diagnostic scalar loss below always reports all three regardless.
fim_objectives = ("E", "F", "Q")
_valid_objectives = {"E", "F", "Q"}
if not fim_objectives or not set(fim_objectives).issubset(_valid_objectives):
    raise ValueError(f"fim_objectives must be a non-empty subset of {_valid_objectives}; got {fim_objectives}")
fim_objectives = tuple(o for o in ("E", "F", "Q") if o in fim_objectives)  # canonical order

wE = 1.0
wF = 1.0
wQ = 1.0
eps = 1e-12
penalty_value = 1e6

calculate_L = True
calculate_FIM = True
finite_difference_h = 1e-2
finite_difference_method = "central"
run_directional_scans = True
scan_amplitudes = np.linspace(-0.5, 0.5, 9)
scan_mode_ids = [0, 1, 2, 3, 4, 8]  # "auto" = stiff, middle, sloppiest; or list of ints

print(f"Dataset path: {dataset_path}")
print(f"DFT (revpbe0) out dir: {dft_out_dir}")
print(f"Reference charge fchk dir: {fchk_dir}")
print(f"Reference charge cache dir: {chg_root}")
print(f"MAX_STRUCTURES_FOR_TEST: {MAX_STRUCTURES_FOR_TEST}")
print(f"FIM objectives: {fim_objectives}")
print(f"Loss component weights: wE={wE}, wF={wF}, wQ={wQ}")
print(f"calculate_L: {calculate_L}, calculate_FIM: {calculate_FIM}")


# ----------------------- parameters ----------------------- #

param_changes = {
    "kpair": {
        "Na-O": 1.0,
        "Na-H": 1.0,
    },
    "Na": {
        "levels": [-4.5469340000000003, -1.3327190000000000],
        "slater": [0.7637870000000000, 0.5735530000000000],
        "gam": 0.2710560000000000,
        "lgam": [1.0000000000000000, 1.1018893999999999],
        "gam3": 0.1798727000000000,
        "zeff": 5.2449170000000001,
        "arep": 0.5727280000000000,
    },
}

fim_param_changes = param_changes  # Na-only; H/O fixed externally via FIXED_HO_TOML_PATH




fixed_param_changes = load_fixed_params_from_toml(FIXED_HO_TOML_PATH, FIXED_ELEMENT_PARAMS, FIXED_KPAIR_KEYS)
print(f"Fixed H/O parameter file: {FIXED_HO_TOML_PATH}")
print("Fixed H/O and water kpair parameters:")
print(fixed_param_changes)



names, bounds, default_gfn2_vector, structure = generate_parameter_info(
    fim_param_changes,
    fixed_components={("Na", "levels", 1), ("Na", "slater", 1)},
)
param_scale = 0.5 * (np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0])
print("Parameter names:")
print(names)
print("Bounds:")
print(bounds)
print(f"Number of included parameters {len(names)}")


# ----------------------- dataset: select Na-relevant tiers ----------------------- #

coord_files = []
pattern = re.compile(r'^\d+_.*\.xyz$')
for filename in os.listdir(dataset_path):
    match = pattern.match(filename)
    if match:
        file_number = int(match.group().split('_')[0])
        coord_files.append((file_number, filename))
coord_files.sort(key=lambda x: x[0])



na_coord_files = [(n, f) for n, f in coord_files if is_na_tier(n)]
if MAX_STRUCTURES_FOR_TEST is not None:
    na_coord_files = na_coord_files[:MAX_STRUCTURES_FOR_TEST]
print(f"Na-relevant structures selected (T2_Na + T3): {len(na_coord_files)}")


# ----------------------- reference E/F/Q (revpbe0) ----------------------- #

print("Precomputing reference partial charges from revpbe0 .fchk files (cached; runs once per structure)...")
precompute_multiwfn_charges_from_fchk(
    fchk_dir=fchk_dir,
    xyz_dir=dataset_path,
    chg_root=chg_root,
    multiwfn_exe=MULTIWFN_EXE,
    atmrad_dir=ATMRAD_DIR,
    schemes=(type_charge,),
    stems={Path(filename).stem for _, filename in na_coord_files},
    max_workers=N_WORKERS,
    omp_multiwfn=MULTIWFN_THREADS,
    tmp_root=MULTIWFN_REF_PRECOMPUTE_DIR,
    overwrite=False,
    show_progress=True,
)

res_ref, forces_ref, _, charges_ref = load_dft_reference_dataset_with_scheme(
    coord_files=na_coord_files,
    dataset_path=dataset_path,
    dft_out_dir=dft_out_dir,
    chg_root=chg_root,
    scheme=type_charge,
    monomer_out_dir=monomer_out_dir,
)
if use_na_o_target:
    na_o_target_files = [(int(pattern.match(f).group().split('_')[0]), f)
                          for f in sorted(os.listdir(NA_O_TARGET_DIR)) if pattern.match(f)]
    print(f"use_na_o_target=True: adding {len(na_o_target_files)} Na-O target structures: {[f for _, f in na_o_target_files]}")
    precompute_multiwfn_charges_from_fchk(
        fchk_dir=NA_O_TARGET_FCHK_DIR,
        xyz_dir=NA_O_TARGET_DIR,
        chg_root=NA_O_TARGET_CHG_DIR,
        multiwfn_exe=MULTIWFN_EXE,
        atmrad_dir=ATMRAD_DIR,
        schemes=(type_charge,),
        stems={Path(f).stem for _, f in na_o_target_files},
        max_workers=N_WORKERS,
        omp_multiwfn=MULTIWFN_THREADS,
        tmp_root=MULTIWFN_REF_PRECOMPUTE_DIR,
        overwrite=False,
        show_progress=True,
    )
    res_tgt, forces_tgt, _, charges_tgt = load_dft_reference_dataset_with_scheme(
        coord_files=na_o_target_files,
        dataset_path=NA_O_TARGET_DIR,
        dft_out_dir=NA_O_TARGET_DFT_DIR,
        chg_root=NA_O_TARGET_CHG_DIR,
        scheme=type_charge,
        monomer_out_dir=monomer_out_dir,
    )
    na_coord_files = na_coord_files + [(n, f"na_o_target/{f}") for n, f in na_o_target_files]
    res_ref = np.concatenate([res_ref, res_tgt])
    forces_ref = list(forces_ref) + list(forces_tgt)
    charges_ref = list(charges_ref) + list(charges_tgt)

if use_na_h2o4_target:
    na_h2o4_target_files = [(int(pattern.match(f).group().split('_')[0]), f)
                             for f in sorted(os.listdir(NA_H2O4_TARGET_DIR)) if pattern.match(f)]
    print(f"use_na_h2o4_target=True: adding {len(na_h2o4_target_files)} Na(H2O)4 target structures: {[f for _, f in na_h2o4_target_files]}")
    precompute_multiwfn_charges_from_fchk(
        fchk_dir=NA_H2O4_TARGET_FCHK_DIR,
        xyz_dir=NA_H2O4_TARGET_DIR,
        chg_root=NA_H2O4_TARGET_CHG_DIR,
        multiwfn_exe=MULTIWFN_EXE,
        atmrad_dir=ATMRAD_DIR,
        schemes=(type_charge,),
        stems={Path(f).stem for _, f in na_h2o4_target_files},
        max_workers=N_WORKERS,
        omp_multiwfn=MULTIWFN_THREADS,
        tmp_root=MULTIWFN_REF_PRECOMPUTE_DIR,
        overwrite=False,
        show_progress=True,
    )
    res_h2o4_tgt, forces_h2o4_tgt, _, charges_h2o4_tgt = load_dft_reference_dataset_with_scheme(
        coord_files=na_h2o4_target_files,
        dataset_path=NA_H2O4_TARGET_DIR,
        dft_out_dir=NA_H2O4_TARGET_DFT_DIR,
        chg_root=NA_H2O4_TARGET_CHG_DIR,
        scheme=type_charge,
        monomer_out_dir=monomer_out_dir,
    )
    na_coord_files = na_coord_files + [(n, f"na_h2o4_target/{f}") for n, f in na_h2o4_target_files]
    res_ref = np.concatenate([res_ref, res_h2o4_tgt])
    forces_ref = list(forces_ref) + list(forces_h2o4_tgt)
    charges_ref = list(charges_ref) + list(charges_h2o4_tgt)

N_atoms_ref = np.array([np.asarray(forces_ref[i]).shape[0] for i in range(len(res_ref))], dtype=float)
if np.any(N_atoms_ref == 0):
    raise ValueError("Encountered structure with zero atoms.")

refs = (res_ref, forces_ref, charges_ref)
print(f"Number of FIM evaluation structures: {len(na_coord_files)}")


# ----------------------- calculator (tblite + Multiwfn) ----------------------- #

template_cache = (
    prepare_molden_templates_gfn2(
        coord_files=na_coord_files,
        dataset_path=dataset_path,
        template_dir=TEMPLATE_DIR,
        xtb_exe=os.environ.get("XTB_EXE", "xtb"),
        overwrite=False,
    )
    if PREPARE_TEMPLATES
    else None
)

if template_cache is None:
    if not TEMPLATE_DIR.exists():
        raise RuntimeError(
            f"Molden template directory does not exist: {TEMPLATE_DIR}. "
            "Run once with --prepare-templates after changing the dataset."
        )
    template_cache = MoldenTemplateCache()
    missing = []
    for _, filename in na_coord_files:
        xyz_path = dataset_path / filename
        charge = get_total_charge(xyz_path)
        templ_path = TEMPLATE_DIR / f"{xyz_path.stem}__q{charge:+d}.molden"
        if not templ_path.exists():
            missing.append(str(templ_path))
            continue
        template_cache.add(xyz_path.stem, charge, templ_path)
    if missing:
        preview = "\n".join(missing[:10])
        more = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise RuntimeError(
            "Missing Molden templates for the current dataset. "
            f"Run once with --prepare-templates after changing the dataset.\n{preview}{more}"
        )

MONO_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/monomers"
monomer_na_xyz = MONO_DIR / "na/na.xyz"
monomer_cl_xyz = MONO_DIR / "cl/cl.xyz"
monomer_wat_xyz = MONO_DIR / "water/water.xyz"


def calculate(vector, coord_files, type_charge=type_charge, print_timing=False):
    optimized_changes = vector_to_param_changes(vector, structure)
    full_param_changes = merge_param_changes(fixed_param_changes, optimized_changes)
    res, forces, charges = calculate_dict_tblite_multiwfn_parallel_charges(
        param_changes=full_param_changes,
        coord_files=coord_files,
        dataset_path=dataset_path,
        template_cache=template_cache,
        tblite_root=TBLITE_RUNS,
        molden_out_dir=MOLDEN_OUT,
        multiwfn_root=MULTIWFN_ROOT,
        multiwfn_exe=MULTIWFN_EXE,
        atmrad_dir=ATMRAD_DIR,
        scheme=type_charge,
        max_workers=N_WORKERS,
        omp_multiwfn=MULTIWFN_THREADS,
        iterations=2000,
        restart_policy="recreate",
        monomer_na_xyz=monomer_na_xyz,
        monomer_cl_xyz=monomer_cl_xyz,
        monomer_wat_xyz=monomer_wat_xyz,
        MONO_DIR=MONOMER_RUN_DIR,
        compute_relative_energy=True,
        WORKDIR=WORKDIR,
        clean_multiwfn_root=False,
        show_progress=False,
        print_timing=print_timing,
    )
    return res, forces, charges


run_config = {
    "run_id": RUN_ID,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_cpus_per_task": HPC.slurm_cpus,
    "n_workers": N_WORKERS,
    "multiwfn_threads": MULTIWFN_THREADS,
    "multiwfn_exe": str(MULTIWFN_EXE),
    "scratch_dir": str(SCRATCH_DIR),
    "workdir": str(WORKDIR),
    "output_dir": str(RUN_OUTPUT_DIR),
    "dataset_path": str(dataset_path),
    "dft_out_dir": str(dft_out_dir),
    "fchk_dir": str(fchk_dir),
    "chg_root": str(chg_root),
    "use_na_o_target": use_na_o_target,
    "use_na_h2o4_target": use_na_h2o4_target,
    "charge_scheme": type_charge,
    "fixed_ho_toml_path": str(FIXED_HO_TOML_PATH),
    "fixed_param_changes": fixed_param_changes,
    "allowed_tiers": ["T2_Na", "T3"],
    "n_structures": len(na_coord_files),
    "weights": {"E": wE, "F": wF, "Q": wQ},
    "eps": eps,
    "calculate_L": calculate_L,
    "calculate_FIM": calculate_FIM,
    "finite_difference_h": finite_difference_h,
    "finite_difference_method": finite_difference_method,
    "fim_objectives": list(fim_objectives),
    "fim_parameter_names": names,
    "fim_bounds": bounds,
    "fim_param_scale": param_scale.tolist(),
}
(RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


# ----------------------- output helpers ----------------------- #


def make_parity_plots(E_pred, F_pred_list, Q_pred_list, refs=refs):
    E_ref, F_ref_list, Q_ref_list = refs
    ha_to_kcal = 627.5094740631
    parity_plot(
        np.asarray(E_ref) * ha_to_kcal,
        np.asarray(E_pred) * ha_to_kcal,
        "Reference relative energy (kcal/mol)",
        "GFN2-xTB relative energy (kcal/mol)",
        "Energy parity",
        RUN_OUTPUT_DIR / "parity_energy_kcalmol.png",
    )
    force_ha_bohr_to_mev_a = 1000.0 * 27.211386245988 / 0.529177210903
    parity_plot(
        np.concatenate([np.asarray(f).ravel() for f in F_ref_list]) * force_ha_bohr_to_mev_a,
        np.concatenate([np.asarray(f).ravel() for f in F_pred_list]) * force_ha_bohr_to_mev_a,
        "Reference force (meV/A)",
        "GFN2-xTB force (meV/A)",
        "Force parity",
        RUN_OUTPUT_DIR / "parity_forces_mevA.png",
    )
    parity_plot(
        np.concatenate([np.asarray(q).ravel() for q in Q_ref_list]),
        np.concatenate([np.asarray(q).ravel() for q in Q_pred_list]),
        f"Reference {type_charge} charge (e)",
        "GFN2-xTB charge (e)",
        "Partial charge parity",
        RUN_OUTPUT_DIR / "parity_charges_e.png",
    )


# ----------------------- loss definition (diagnostic only, always full E/F/Q) ----------------------- #

def loss(vector, coord_files=na_coord_files, refs=refs, N_atoms=N_atoms_ref, wE=wE, wF=wF, wQ=wQ, penalty_value=penalty_value, eps=eps, return_components=False, print_components=False):
    """
    L = wE * [ Sum_i (1/N_i) ((dE_i - <dE>)^2) / Sum_i (1/N_i) (E_ref_i - <E_ref>)^2 ]
      + wF * [ Sum_i (1/(3N_i)) ||dF_i||^2 / Sum_i (1/(3N_i)) ||F_ref_i||^2 ]
      + wQ * [ Sum_i (1/N_i) Sum_a (dq_ia)^2 / Sum_i (1/N_i) Sum_a (q_ref_ia)^2 ]
    Always computed in full (E, F, Q), independent of fim_objectives -- this is
    the diagnostic scalar loss, not the FIM residual.
    """
    E_ref, F_ref_list, Q_ref_list = refs

    try:
        E_pred, F_pred_list, Q_pred_list = calculate(vector, coord_files)
    except Exception as e:
        components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
        return (penalty_value, components) if return_components else penalty_value

    n = len(E_ref)
    if not (len(E_pred) == len(F_ref_list) == len(F_pred_list) == len(Q_ref_list) == len(Q_pred_list) == n):
        components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
        return (penalty_value, components) if return_components else penalty_value

    dE = np.asarray(E_pred, dtype=float) - np.asarray(E_ref, dtype=float)
    wE_i = 1.0 / N_atoms
    sum_wE = np.sum(wE_i)
    mean_dE = np.sum(wE_i * dE) / sum_wE
    mean_Eref = np.sum(wE_i * np.asarray(E_ref, dtype=float)) / sum_wE
    E_numerator = np.sum(wE_i * (dE - mean_dE) ** 2)
    E_denominator = np.sum(wE_i * (np.asarray(E_ref, dtype=float) - mean_Eref) ** 2)

    F_numerator = 0.0
    F_denominator = 0.0
    Q_numerator = 0.0
    Q_denominator = 0.0
    for i in range(n):
        Fref = np.asarray(F_ref_list[i], dtype=float)
        Fpred = np.asarray(F_pred_list[i], dtype=float)
        if Fref.shape != Fpred.shape or Fref.ndim != 2 or Fref.shape[1] != 3:
            components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
            return (penalty_value, components) if return_components else penalty_value
        Ni = Fref.shape[0]
        dF = Fpred - Fref
        wF_i = 1.0 / (3.0 * Ni)
        F_numerator += wF_i * float(np.sum(dF * dF))
        F_denominator += wF_i * float(np.sum(Fref * Fref))

        qref = np.asarray(Q_ref_list[i], dtype=float).ravel()
        qpred = np.asarray(Q_pred_list[i], dtype=float).ravel()
        if qref.size != qpred.size:
            components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
            return (penalty_value, components) if return_components else penalty_value
        wQa_i = 1.0 / qref.size
        dq = qpred - qref
        Q_numerator += wQa_i * float(np.sum(dq * dq))
        Q_denominator += wQa_i * float(np.sum(qref * qref))

    if not np.isfinite(E_denominator) or E_denominator < eps:
        components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
        return (penalty_value, components) if return_components else penalty_value
    if not np.isfinite(F_denominator) or F_denominator < eps:
        components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
        return (penalty_value, components) if return_components else penalty_value
    if not np.isfinite(Q_denominator) or Q_denominator < eps:
        components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
        return (penalty_value, components) if return_components else penalty_value

    E_term = E_numerator / (E_denominator + eps)
    F_term = F_numerator / (F_denominator + eps)
    Q_term = Q_numerator / (Q_denominator + eps)
    L = wE * E_term + wF * F_term + wQ * Q_term
    if not np.isfinite(L):
        components = {"E": penalty_value, "F": penalty_value, "Q": penalty_value}
        return (penalty_value, components) if return_components else penalty_value

    if print_components:
        print(f"Loss: {L:.10f} (dE: {E_term:.10f}, dF: {F_term:.10f}, dQ: {Q_term:.10f})")
        make_parity_plots(E_pred, F_pred_list, Q_pred_list, refs=refs)

    components = {"E": float(E_term), "F": float(F_term), "Q": float(Q_term)}
    return (float(L), components) if return_components else float(L)


# ----------------------- FIM / Gauss-Newton helpers (objective-selectable) ----------------------- #

@dataclass
class ResidualNormCache:
    denominators: dict = field(default_factory=dict)  # objective -> normalization denominator
    wE_i: np.ndarray = None


def make_residual_norm_cache(E_ref, N_atoms, F_ref_list, Q_ref_list, fim_objectives=fim_objectives, eps=eps):
    E_ref = np.asarray(E_ref, dtype=float)
    wE_i = 1.0 / N_atoms
    denominators = {}

    if "E" in fim_objectives:
        mean_Eref = np.sum(wE_i * E_ref) / np.sum(wE_i)
        denominators["E"] = float(np.sum(wE_i * (E_ref - mean_Eref) ** 2))

    if "F" in fim_objectives:
        F_denominator = 0.0
        for Fref in F_ref_list:
            Fref = np.asarray(Fref, dtype=float)
            Ni = Fref.shape[0]
            F_denominator += (1.0 / (3.0 * Ni)) * float(np.sum(Fref * Fref))
        denominators["F"] = float(F_denominator)

    if "Q" in fim_objectives:
        Q_denominator = 0.0
        for qref in Q_ref_list:
            qref = np.asarray(qref, dtype=float).ravel()
            Q_denominator += (1.0 / qref.size) * float(np.sum(qref * qref))
        denominators["Q"] = float(Q_denominator)

    for obj, denom in denominators.items():
        if not np.isfinite(denom) or denom < eps:
            raise ValueError(f"Bad {obj}_denominator: {denom}")

    return ResidualNormCache(denominators=denominators, wE_i=wE_i)


_calculate_cache = {}



def calculate_cached(vector, coord_files=na_coord_files, use_cache=True):
    key = (_vector_cache_key(vector), tuple(f for _, f in coord_files))
    if use_cache and key in _calculate_cache:
        return _calculate_cache[key]
    out = calculate(vector, coord_files)
    if use_cache:
        _calculate_cache[key] = out
    return out


def residual_vector(vector, coord_files=na_coord_files, refs=refs, wE=wE, wF=wF, wQ=wQ, eps=eps, norm_cache=None, return_slices=False, fim_objectives=fim_objectives):
    """Build r such that dot(r, r) reproduces the scalar cluster loss restricted to the
    objectives in fim_objectives (a subset of {E, F, Q})."""
    weights = {"E": wE, "F": wF, "Q": wQ}
    E_ref, F_ref_list, Q_ref_list = refs
    E_pred, F_pred_list, Q_pred_list = calculate_cached(vector, coord_files)

    if norm_cache is None:
        norm_cache = make_residual_norm_cache(E_ref, N_atoms_ref, F_ref_list, Q_ref_list, fim_objectives=fim_objectives, eps=eps)

    parts = {}

    if "E" in fim_objectives:
        E_ref_arr = np.asarray(E_ref, dtype=float)
        dE = np.asarray(E_pred, dtype=float) - E_ref_arr
        mean_dE = np.sum(norm_cache.wE_i * dE) / np.sum(norm_cache.wE_i)
        parts["E"] = np.sqrt(weights["E"] * norm_cache.wE_i / (norm_cache.denominators["E"] + eps)) * (dE - mean_dE)

    if "F" in fim_objectives:
        rF_parts = []
        for i in range(len(F_ref_list)):
            Fref = np.asarray(F_ref_list[i], dtype=float)
            Fpred = np.asarray(F_pred_list[i], dtype=float)
            Ni = Fref.shape[0]
            rF_parts.append(
                (np.sqrt(weights["F"] * (1.0 / (3.0 * Ni)) / (norm_cache.denominators["F"] + eps)) * (Fpred - Fref)).reshape(-1)
            )
        parts["F"] = np.concatenate(rF_parts, axis=0)

    if "Q" in fim_objectives:
        rQ_parts = []
        for i in range(len(Q_ref_list)):
            qref = np.asarray(Q_ref_list[i], dtype=float).ravel()
            qpred = np.asarray(Q_pred_list[i], dtype=float).ravel()
            rQ_parts.append(
                np.sqrt(weights["Q"] * (1.0 / qref.size) / (norm_cache.denominators["Q"] + eps)) * (qpred - qref)
            )
        parts["Q"] = np.concatenate(rQ_parts, axis=0)

    r = np.concatenate([parts[obj] for obj in fim_objectives], axis=0)

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
                "block": block,
                "mode": mode,
                "eigenvalue": float(lam),
                "relative_to_max": float(lam / evals[0]) if evals[0] != 0 else np.nan,
                "cumulative_trace_fraction": float(np.sum(np.clip(evals[: mode + 1], 0.0, None)) / total) if total > 0 else np.nan,
            })
    write_csv_rows(
        RUN_OUTPUT_DIR / "fim_eigenvalues.csv",
        rows,
        ["block", "mode", "eigenvalue", "relative_to_max", "cumulative_trace_fraction"],
    )


def save_eigenvector_csv(evals, evecs):
    rows = []
    for mode, lam in enumerate(evals):
        for pidx, name in enumerate(names):
            coeff = float(evecs[pidx, mode])
            rows.append({
                "mode": mode,
                "eigenvalue": float(lam),
                "parameter": name,
                "coefficient": coeff,
                "participation": coeff * coeff,
            })
    write_csv_rows(
        RUN_OUTPUT_DIR / "fim_total_eigenvectors.csv",
        rows,
        ["mode", "eigenvalue", "parameter", "coefficient", "participation"],
    )



def plot_mode_participation_heatmap(evecs, mode_ids, out_path, title="Participation factor for all modes"):
    V = np.asarray(evecs[:, mode_ids], dtype=float)
    P = V ** 2
    P_masked = np.ma.masked_where(P <= 0.0, P)

    fig_width = max(8.0, 0.35 * len(mode_ids) + 4.0)
    fig_height = max(5.0, 0.35 * len(names) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(P_masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(mode_ids)))
    ax.set_xticklabels([f"mode_{mode}" for mode in mode_ids], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="v_jk^2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
    modes = [row["mode"] for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(9.0, 0.35 * len(rows) + 4.0), 4.5))
    bottom = np.zeros(len(rows))
    for obj in fim_objectives:
        frac = np.asarray([row[f"{obj}_frac"] for row in rows], dtype=float)
        ax.bar(x, frac, bottom=bottom, label=obj)
        bottom = bottom + frac
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_xlabel("Total mode")
    ax.set_ylabel("Fraction of lambda_total")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scan_along_eigendirection(theta_ref, vec_scaled, amplitudes, scale, lower, upper):
    rows = []
    for amp in amplitudes:
        theta = np.asarray(theta_ref, dtype=float) + amp * (scale * vec_scaled)
        hit_lower = bool(np.any(theta < lower))
        hit_upper = bool(np.any(theta > upper))
        theta = np.minimum(np.maximum(theta, lower), upper)
        L, comps = loss(theta, return_components=True, print_components=False)
        rows.append({
            "amp": float(amp),
            "L": float(L),
            "E": float(comps["E"]),
            "F": float(comps["F"]),
            "Q": float(comps["Q"]),
            "hit_lower": hit_lower,
            "hit_upper": hit_upper,
        })
    return rows


def plot_directional_scan(rows, lambda_k, out_path, title):
    amp = np.asarray([row["amp"] for row in rows], dtype=float)
    L = np.asarray([row["L"] for row in rows], dtype=float)
    E = np.asarray([row["E"] for row in rows], dtype=float)
    F = np.asarray([row["F"] for row in rows], dtype=float)
    Q = np.asarray([row["Q"] for row in rows], dtype=float)
    zero_idx = int(np.argmin(np.abs(amp)))
    L0 = L[zero_idx]
    L_quad = L0 + lambda_k * amp ** 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(amp, L, marker="o", label="actual total loss")
    axes[0].plot(amp, L_quad, linestyle="--", label="quadratic L0 + lambda_k a^2")
    axes[0].set_xlabel("Amplitude along eigendirection")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(amp, E, marker="o", label="E")
    axes[1].plot(amp, F, marker="o", label="F")
    axes[1].plot(amp, Q, marker="o", label="Q")
    axes[1].set_xlabel("Amplitude")
    axes[1].set_ylabel("Component value")
    axes[1].set_title(title + " components")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_fim_analysis():
    theta_ref = np.asarray(default_gfn2_vector, dtype=float)
    bounds_arr = np.asarray(bounds, dtype=float)
    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    scale = np.asarray(param_scale, dtype=float)

    norm_cache = make_residual_norm_cache(res_ref, N_atoms_ref, forces_ref, charges_ref, fim_objectives=fim_objectives)
    r_ref, slices_ref, parts_ref = residual_vector(
        theta_ref,
        norm_cache=norm_cache,
        return_slices=True,
        fim_objectives=fim_objectives,
    )

    L0_from_r = float(np.dot(r_ref, r_ref))
    weighted_residual_block_terms = {key: float(np.dot(value, value)) for key, value in parts_ref.items()}
    print("Residual sizes:")
    for obj in fim_objectives:
        print(f"  {obj}:", parts_ref[obj].size)
    print("  total:", r_ref.size)
    print("||r||^2 =", L0_from_r)
    print("Weighted residual block contributions:", weighted_residual_block_terms)

    def residual_func_for_fim(theta_raw):
        return residual_vector(theta_raw, norm_cache=norm_cache, return_slices=False, fim_objectives=fim_objectives)

    start_time = time.time()
    Jz, r0, steps_used, fd_mode = compute_jacobian_fd_scaled(
        theta_ref=theta_ref,
        residual_func=residual_func_for_fim,
        scale=scale,
        lower=lower,
        upper=upper,
        h=finite_difference_h,
        method=finite_difference_method,
        verbose=True,
    )
    fim_time = time.time() - start_time

    fim_blocks = compute_fim_from_jacobian(Jz, slices=slices_ref)
    evals_total, evecs_total = eigh_sorted(fim_blocks["total"], descending=True)
    evals_by_obj = {}
    evecs_by_obj = {}
    for obj in fim_objectives:
        evals_by_obj[obj], evecs_by_obj[obj] = eigh_sorted(fim_blocks[obj], descending=True)

    print("\nSteps actually used:")
    for name, h_j, mode in zip(names, steps_used, fd_mode):
        print(f"{name:20s}  h={h_j:.5e}  mode={mode}")

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
        Jz=Jz,
        r0=r0,
        steps_used=steps_used,
        fd_mode=np.asarray(fd_mode),
        evals_total=evals_total,
        evecs_total=evecs_total,
        F_total=fim_blocks["total"],
        names=np.asarray(names),
        theta_ref=theta_ref,
        scale=scale,
        lower=lower,
        upper=upper,
        fim_objectives=np.asarray(fim_objectives),
        **{f"evals_{obj}": evals_by_obj[obj] for obj in fim_objectives},
        **{f"evecs_{obj}": evecs_by_obj[obj] for obj in fim_objectives},
        **{f"F_{obj}": fim_blocks[obj] for obj in fim_objectives},
    )

    save_eigenvalue_csv({"total": evals_total, **evals_by_obj})
    save_eigenvector_csv(evals_total, evecs_total)
    write_csv_rows(
        RUN_OUTPUT_DIR / "fim_top_loadings.csv",
        loading_rows,
        ["which", "mode", "eigenvalue", "rank", "parameter", "coefficient", "participation"],
    )

    plot_spectrum(evals_total, "total", RUN_OUTPUT_DIR / "fim_spectrum_total.png")
    spectrum_label = {"E": "energy", "F": "forces", "Q": "charges"}
    for obj in fim_objectives:
        plot_spectrum(evals_by_obj[obj], spectrum_label[obj], RUN_OUTPUT_DIR / f"fim_spectrum_{spectrum_label[obj]}.png")

    mode_ids = list(range(len(evals_total)))[::-1]
    plot_mode_participation_heatmap(
        evecs_total,
        mode_ids,
        RUN_OUTPUT_DIR / "fim_participation_heatmap_all_modes.png",
        title="Participation factor for all modes (sloppy to stiff)",
    )

    block_rows = block_contributions_for_total_modes(fim_blocks, evals_total, evecs_total, fim_objectives=fim_objectives)
    block_fields = ["mode", "lambda_total"]
    for obj in fim_objectives:
        block_fields.extend([f"{obj}_abs", f"{obj}_frac"])
    write_csv_rows(
        RUN_OUTPUT_DIR / "fim_block_contribution_fractions.csv",
        block_rows,
        block_fields,
    )
    plot_block_fractions(
        block_rows,
        RUN_OUTPUT_DIR / "fim_block_contribution_fractions_all_modes.png",
        fim_objectives=fim_objectives,
        title="Block contribution fractions for all modes",
    )

    scan_rows_all = []
    if run_directional_scans:
        if scan_mode_ids == "auto":
            selected_modes = sorted(set([0, len(evals_total) // 2, len(evals_total) - 1]))
        else:
            selected_modes = list(scan_mode_ids)
        for mode in selected_modes:
            rows = scan_along_eigendirection(theta_ref, evecs_total[:, mode], scan_amplitudes, scale, lower, upper)
            for row in rows:
                row["mode"] = int(mode)
                row["lambda_total"] = float(evals_total[mode])
            scan_rows_all.extend(rows)
            plot_directional_scan(
                rows,
                float(evals_total[mode]),
                RUN_OUTPUT_DIR / f"fim_directional_scan_mode_{mode:03d}.png",
                title=f"Mode {mode} scan",
            )
        write_csv_rows(
            RUN_OUTPUT_DIR / "fim_directional_scans.csv",
            scan_rows_all,
            ["mode", "lambda_total", "amp", "L", "E", "F", "Q", "hit_lower", "hit_upper"],
        )

    summary = {
        "run_id": RUN_ID,
        "n_structures_selected": len(na_coord_files),
        "n_parameters": len(names),
        "parameter_names": names,
        "parameter_scope": "Na element parameters plus Na-O, Na-H kpair terms (H/O zeff/arep and O-O/O-H kpair fixed to optimised cluster water values)",
        "charge_scheme": type_charge,
        "fim_objectives": list(fim_objectives),
        "theta_ref_label": "default_gfn2_Na",
        "finite_difference_h": finite_difference_h,
        "finite_difference_method": finite_difference_method,
        "steps_used": steps_used.tolist(),
        "fd_mode": fd_mode,
        "residual_sizes": {key: int(value.size) for key, value in parts_ref.items()},
        "residual_size_total": int(r_ref.size),
        "loss_from_residual_dot": L0_from_r,
        "weighted_loss_blocks_from_residual_dot": weighted_residual_block_terms,
        "fim_wall_time_seconds": float(fim_time),
        "condition_info": {
            "total": fim_condition_info(evals_total),
            **{obj: fim_condition_info(evals_by_obj[obj]) for obj in fim_objectives},
        },
        "outputs": {
            "npz": "fim_results_default_gfn2.npz",
            "eigenvalues": "fim_eigenvalues.csv",
            "eigenvectors": "fim_total_eigenvectors.csv",
            "top_loadings": "fim_top_loadings.csv",
            "participation_heatmap": "fim_participation_heatmap_all_modes.png",
            "block_fractions": "fim_block_contribution_fractions_all_modes.png",
        },
    }
    (RUN_OUTPUT_DIR / "fim_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ----------------------- optional run blocks ----------------------- #

if calculate_L:
    print("Running reference loss calculation before FIM...")
    start_time = time.time()
    L = loss(default_gfn2_vector, print_components=True)
    sweep_time = time.time() - start_time
    print(f"Sweep time: {sweep_time:.2f} seconds")
    print(L)

if calculate_FIM:
    print("Running cluster Na FIM analysis (selectable objectives)...")
    run_fim_analysis()
