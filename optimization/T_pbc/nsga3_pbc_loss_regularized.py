"""
NSGA-III optimization of the PBC water tblite loss.

This file is intentionally written in the same style as the existing NSGA
scripts: settings are edited directly in the script, and CLI arguments are only
for HPC/run plumbing.

The optimized objectives are the normalized PBC loss components for fitted
composition-offset energy residuals, forces, and virials, restricted to a
selectable subset via a single keyword (same pattern as `fim_objectives` in
pbc_fim_water_select_objectives.py):

  objectives = ("E", "F", "Xi")   # full set (default, matches the original script)
  objectives = ("E", "F")         # drop virials
  objectives = ("F",)             # forces-only
  ...

By default the script uses a fixed random 70/30 water split and optimizes on
the 210-structure train split. To optimize all current water structures, set:

  split_dataset = False

A full-parameter regularization objective penalizes deviation of all optimized
parameters from their GFN2-xTB default values.  It is enabled by default:

  use_param_regularization = True

When enabled, the NSGA-III objectives are `objectives` plus Reg, where:

  Reg = mean_i( ((theta_i - theta0_i) / param_scale_i)^2 )

with param_scale_i = 0.5*(upper_i - lower_i).  Value is 0 at the default
point and at most 1 when every parameter is at its bound.

Each generation writes a preliminary feasible non-dominated front to:

  <output_root>/<run_id>/generation_fronts/

This makes it possible to inspect the current Pareto front while the job is
still running.

Usage:

  python -u optimization/T_pbc/nsga3_pbc_loss_regularized.py --run-id nsga3_pbc_smoke

For a small test keep:

  MAX_STRUCTURES_FOR_TEST = 3
  calculate_L = True
  run_optimization = True
  direction_number = 8 + 1
  n_gen = 1

For the full train-split optimization set:

  MAX_STRUCTURES_FOR_TEST = None
  system_filter = "water"
  split_dataset = True
  direction_number = 512 + 1
  n_gen = 40

Submit on the cluster with:

  sbatch optimization/T_pbc/run_nsga3_pbc_loss_regularized.slurm

Default energy residual convention:

  Eres_i = E_tot_i - sum_A n_i,A * eps_A

where eps_A are fitted by least squares for the selected successful structures.
The legacy isolated-atom atomization-energy path remains available by setting
energy_mode = "isolated_atoms" below.

Internal loss units:

  energy residuals: Ha
  forces: Ha/Bohr
  virials: Ha, 3x3 tensor
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.environ.get('USER', 'user')}"))

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import qmc

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import ensure_param_file, get_base_param_file
from codes.pbc_data import (
    GPA_A3_TO_HA,
    REFERENCE_VIRIAL_SIGN,
    atomization_energy,
    ensure_poscar_cache,
    fit_composition_offsets,
    load_pbc_dataset,
    load_reference_atom_energies,
)
from codes.pbc_tblite import get_or_run_isolated_atom_energies, run_tblite_pbc
from codes.common import (
    init_tblite_env,
    generate_parameter_info,
    generate_sobol_initial_population,
    nondominated_indices,
    parity_plot,
    force_component_parity_plot,
)


# ----------------------- runtime setup ----------------------- #


init_tblite_env(omp_threads=1, blas_threads=1, amd_mkl=True)


def _default_scratch_root():
    candidates = [Path(p) for p in (os.environ.get("SCRATCH"), "/tmp") if p]
    for path in candidates:
        if path.exists() and os.access(path, os.W_OK):
            return path
    return Path("/tmp")


DEFAULT_SCRATCH_ROOT = _default_scratch_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_PATH / "optimization/T_pbc/hpc_runs"


def _default_run_id():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slurm_job_id:
        return f"nsga3_pbc_loss_{slurm_job_id}_{stamp}"
    return f"nsga3_pbc_loss_local_{stamp}"


def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Run PBC tblite NSGA-III optimization safely on HPC.")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", _default_run_id()))
    parser.add_argument("--scratch-root", default=os.environ.get("SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT)))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    parser.add_argument("--n-workers", type=int, default=None)
    args, _unknown = parser.parse_known_args()

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    env_workers = os.environ.get("N_WORKERS")
    requested_workers = args.n_workers if args.n_workers is not None else (int(env_workers) if env_workers else None)
    args.n_workers = requested_workers or max(1, slurm_cpus)
    args.slurm_cpus = slurm_cpus
    return args


HPC = parse_hpc_args()
RUN_ID = HPC.run_id
SCRATCH_DIR = Path(HPC.scratch_root) / "optxTB-MD" / RUN_ID
WORKDIR = SCRATCH_DIR / "work"
RUN_OUTPUT_DIR = Path(HPC.output_root) / RUN_ID
TBLITE_RUNS = WORKDIR / "pbc_tblite_runs"
ATOM_RUNS = WORKDIR / "isolated_atoms_tblite"
PROPERTIES_DIR = RUN_OUTPUT_DIR / "properties"
GENERATION_FRONTS_DIR = RUN_OUTPUT_DIR / "generation_fronts"
POSCAR_CACHE_DIR = PROJECT_ROOT_PATH / "structures/periodic_data/poscar_cache"

WORKDIR.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROPERTIES_DIR.mkdir(parents=True, exist_ok=True)
GENERATION_FRONTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"RUN_ID: {RUN_ID}")
print(f"Scratch workdir: {WORKDIR}")
print(f"Persistent output dir: {RUN_OUTPUT_DIR}")
print(f"Python workers: {HPC.n_workers}")


# ----------------------- scientific settings ----------------------- #

dataset_path = PROJECT_ROOT_PATH / "structures/periodic_data/dataset"
isolated_atoms_path = PROJECT_ROOT_PATH / "structures/periodic_data/isolated_atoms"
base_param_file = get_base_param_file()

system_filter = "water"  # switch to "all" later after Na reference handling is ready
MAX_STRUCTURES_FOR_TEST = None
split_dataset = True
train_frac = 0.70
split_seed = 123
energy_mode = "fitted_offsets"  # "fitted_offsets" or "isolated_atoms"
fit_elements = ("H", "O", "Na", "Cl")

wE = 0.2
wF = 0.5
wXi = 0.3
eps = 1e-12
penalty_value = 1e6

tblite_iterations = 2000
tblite_acc = 0.0001
tblite_threads = int(os.environ.get("TBLITE_THREADS", "1"))

calculate_L = True
run_optimization = True
use_param_regularization = True

# Keyword for NSGA-III objective choice: any non-empty subset of {"E", "F", "Xi"}.
# Same pattern as fim_objectives in pbc_fim_water_select_objectives.py.
objectives = ("E", "F", "Xi")
_valid_objectives = {"E", "F", "Xi"}
if not objectives or not set(objectives).issubset(_valid_objectives):
    raise ValueError(f"objectives must be a non-empty subset of {_valid_objectives}; got {objectives}")
objectives = tuple(o for o in ("E", "F", "Xi") if o in objectives)  # canonical order

direction_number = 128 + 1 # 128 + 1
n_gen = 40 #40
sobol_seed = 42
nsga_seed = 42

print(f"Dataset path: {dataset_path}")
print(f"System filter: {system_filter}")
print(f"MAX_STRUCTURES_FOR_TEST: {MAX_STRUCTURES_FOR_TEST}")
print(f"Split dataset: {split_dataset}")
print(f"Train fraction: {train_frac}, split seed: {split_seed}")
print(f"Energy mode: {energy_mode}")
print(f"Fit elements: {fit_elements}")
print(f"Loss component weights: wE={wE}, wF={wF}, wXi={wXi}")
print(f"tblite threads per worker: {tblite_threads}")
print(f"Run optimization: {run_optimization}")
print(f"Use full-parameter regularization objective: {use_param_regularization}")
print(f"Objectives: {objectives}")


# ----------------------- parameters ----------------------- #

param_changes = {
    "kpair": {
#        "H-H": 1.0,
        "O-O": 1.0,
        "O-H": 1.0,
    },
    "H": {
#        "gam": 0.4057710000000000,
#        "lgam": [1.0000000000000000],
#        "gam3": 0.0800000000000000,
        "zeff": 1.1053880000000000,
        "arep": 2.2137169999999999,
    },
    "O": {
#        "gam": 0.4518960000000000,
#        "lgam": [1.0000000000000000, 1.1497020000000000],
#        "gam3": -0.0517134000000000,
        "zeff": 5.7844150000000001,
        "arep": 2.1657120000000001,
    },
}



names, bounds, default_gfn2_vector, structure = generate_parameter_info(
    param_changes,
    fixed_components=None,
)
bounds_arr = np.asarray(bounds, dtype=float)
lower = bounds_arr[:, 0]
upper = bounds_arr[:, 1]
param_scale = 0.5 * (upper - lower)
param_scale[param_scale == 0.0] = 1.0
theta0 = np.asarray(default_gfn2_vector, dtype=float)

_obj_list = list(objectives)
if use_param_regularization:
    _obj_list.append("Reg")
objective_names = _obj_list


def param_regularization(vector, theta0=theta0, scale=param_scale):
    """Normalized L2 deviation of all parameters from their default values.

    Returns mean_i((theta_i - theta0_i)^2 / param_scale_i^2), where
    param_scale_i = 0.5*(upper_i - lower_i).  Value is 0 at the default
    point and at most 1 when every parameter is at its bound.
    """
    v = np.asarray(vector, dtype=float)
    diff = (v - theta0) / scale
    return float(np.mean(diff * diff))


print("Parameter names:")
print(names)
print("Bounds:")
print(bounds)
print(f"Number of included parameters {len(names)}")
print(f"Objective names: {objective_names}")
print(f"Full-parameter regularization: {'enabled' if use_param_regularization else 'disabled'}, n_params={len(names)}")


# ----------------------- dataset load/extraction ----------------------- #

records_found = load_pbc_dataset(dataset_path, system_filter=system_filter, max_structures=None)

if split_dataset:
    if system_filter != "water":
        raise ValueError("The current random train/test split is intended for system_filter='water'.")
    rng = np.random.default_rng(split_seed)
    shuffled_indices = rng.permutation(len(records_found))
    n_train = int(round(train_frac * len(records_found)))
    train_indices = set(int(idx) for idx in shuffled_indices[:n_train])
    test_indices = set(int(idx) for idx in shuffled_indices[n_train:])
    train_records = [record for idx, record in enumerate(records_found) if idx in train_indices]
    test_records = [record for idx, record in enumerate(records_found) if idx in test_indices]
    records = train_records
else:
    train_records = records_found
    test_records = []
    records = records_found

if MAX_STRUCTURES_FOR_TEST is not None:
    records = records[:MAX_STRUCTURES_FOR_TEST]

records = ensure_poscar_cache(records, POSCAR_CACHE_DIR)
records_by_id = {record.frame_id: record for record in records}

dataset_split = {
    "split_dataset": split_dataset,
    "system_filter": system_filter,
    "train_frac": train_frac,
    "split_seed": split_seed,
    "n_found": len(records_found),
    "n_train": len(train_records),
    "n_test": len(test_records),
    "n_selected_for_sweep": len(records),
    "train_frame_ids": [record.frame_id for record in train_records],
    "test_frame_ids": [record.frame_id for record in test_records],
    "selected_frame_ids": [record.frame_id for record in records],
}
(RUN_OUTPUT_DIR / "dataset_split.json").write_text(json.dumps(dataset_split, indent=2), encoding="utf-8")

symbols = sorted({symbol for record in records for symbol in record.counts})
if energy_mode == "isolated_atoms":
    ref_atom_info = load_reference_atom_energies(isolated_atoms_path, symbols, allow_total_fallback=False)
    ref_atom_energies = {symbol: float(info["energy_ha"]) for symbol, info in ref_atom_info.items()}
    (RUN_OUTPUT_DIR / "reference_atom_energies.json").write_text(json.dumps(ref_atom_info, indent=2), encoding="utf-8")
    print(f"Reference atom energies loaded for: {symbols}")
else:
    ref_atom_info = {}
    ref_atom_energies = {}
    print("Reference atom energies skipped: using fitted composition offsets.")

E_ref = np.asarray([record.ref_energy_ha for record in records], dtype=float)
F_ref = [record.ref_forces_ha_bohr for record in records]
Xi_ref = [record.ref_virial_ha for record in records]
N_atoms = np.asarray([record.n_atoms for record in records], dtype=float)

print(f"Structures found after filter: {len(records_found)}")
if split_dataset:
    print(f"Random train/test split: {len(train_records)} train, {len(test_records)} test")
print(f"Structures selected for this sweep: {len(records)}")
print(f"POSCAR cache ready: {POSCAR_CACHE_DIR}")


run_config = {
    "run_id": RUN_ID,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_cpus_per_task": HPC.slurm_cpus,
    "n_workers": HPC.n_workers,
    "tblite_threads": tblite_threads,
    "scratch_dir": str(SCRATCH_DIR),
    "workdir": str(WORKDIR),
    "output_dir": str(RUN_OUTPUT_DIR),
    "tblite_runs": str(TBLITE_RUNS),
    "atom_runs": str(ATOM_RUNS),
    "generation_fronts_dir": str(GENERATION_FRONTS_DIR),
    "poscar_cache_dir": str(POSCAR_CACHE_DIR),
    "dataset_path": str(dataset_path),
    "isolated_atoms_path": str(isolated_atoms_path),
    "base_param_file": str(base_param_file),
    "system_filter": system_filter,
    "max_structures_for_test": MAX_STRUCTURES_FOR_TEST,
    "split_dataset": split_dataset,
    "train_frac": train_frac,
    "split_seed": split_seed,
    "n_train": len(train_records),
    "n_test": len(test_records),
    "energy_mode": energy_mode,
    "fit_elements": list(fit_elements),
    "weights": {"E": wE, "F": wF, "Xi": wXi},
    "eps": eps,
    "run_optimization": run_optimization,
    "direction_number": direction_number,
    "n_gen": n_gen,
    "sobol_seed": sobol_seed,
    "nsga_seed": nsga_seed,
    "objective_names": objective_names,
    "objectives": list(objectives),
    "use_param_regularization": use_param_regularization,
    "parameter_names": names,
    "bounds": bounds,
    "default_gfn2_vector": list(default_gfn2_vector),
    "internal_units": {"energy": "Ha", "energy_residual": "Ha", "forces": "Ha/Bohr", "virial": "Ha"},
    "reference_virial": {
        "formula": "Xi_ref = -stress_GPa * volume_A3 * GPA_A3_TO_HA",
        "sign": REFERENCE_VIRIAL_SIGN,
        "GPA_A3_TO_HA": GPA_A3_TO_HA,
    },
}
(RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


# ----------------------- calculator ----------------------- #

def save_property_npz(row):
    np.savez_compressed(
        PROPERTIES_DIR / f"{row['frame_id']}.npz",
        ref_energy_ha=row["ref_energy_ha"],
        method_energy_ha=row["method_energy_ha"],
        ref_energy_residual_ha=row.get("ref_energy_residual_ha", np.nan),
        method_energy_residual_ha=row.get("method_energy_residual_ha", np.nan),
        ref_ae_ha=row.get("ref_ae_ha", np.nan),
        method_ae_ha=row.get("method_ae_ha", np.nan),
        ref_forces_ha_bohr=row["ref_forces_ha_bohr"],
        method_forces_ha_bohr=row["method_forces_ha_bohr"],
        ref_virial_ha=row["ref_virial_ha"],
        method_virial_ha=row["method_virial_ha"],
    )


def calculate_one_structure(record, param_toml_path):
    try:
        tblite_result = run_tblite_pbc(
            poscar_path=record.poscar_path,
            nat=record.n_atoms,
            param_toml_path=param_toml_path,
            run_dir=TBLITE_RUNS / record.frame_id,
            charge=0,
            iterations=tblite_iterations,
            acc=tblite_acc,
            omp_threads=tblite_threads,
        )
        row = {
            "status": "ok",
            "reason": "",
            "frame_id": record.frame_id,
            "frame_index": record.frame_index,
            "n_atoms": record.n_atoms,
            "counts": record.counts,
            "ref_energy_ha": record.ref_energy_ha,
            "method_energy_ha": tblite_result.energy_ha,
            "ref_forces_ha_bohr": record.ref_forces_ha_bohr,
            "method_forces_ha_bohr": tblite_result.forces_ha_bohr,
            "ref_virial_ha": record.ref_virial_ha,
            "method_virial_ha": tblite_result.virial_ha,
            "poscar_path": str(record.poscar_path),
            "tblite_out_path": str(tblite_result.out_path),
            "tblite_tag_path": str(tblite_result.tag_path),
        }
        return row
    except Exception as exc:
        return {
            "status": "failed",
            "reason": repr(exc),
            "frame_id": record.frame_id,
            "frame_index": record.frame_index,
            "n_atoms": record.n_atoms,
            "counts": record.counts,
            "ref_energy_ha": record.ref_energy_ha,
            "method_energy_ha": np.nan,
            "poscar_path": str(record.poscar_path),
            "tblite_out_path": "",
            "tblite_tag_path": "",
        }


def calculate(vector, coord_records=records, verbose=True):
    param_dict = vector_to_param_changes(vector, structure)
    param_toml_path = ensure_param_file(param_dict, param_file=str(base_param_file), DIR=WORKDIR)
    run_config["param_toml_path"] = str(param_toml_path)
    (RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    if verbose:
        print(f"Generated parameter TOML: {param_toml_path}")

    if energy_mode == "isolated_atoms":
        method_atom_info = get_or_run_isolated_atom_energies(
            symbols=symbols,
            param_toml_path=param_toml_path,
            cache_root=ATOM_RUNS,
            omp_threads=tblite_threads,
        )
        method_atom_energies = {symbol: float(info["energy_ha"]) for symbol, info in method_atom_info.items()}
        (RUN_OUTPUT_DIR / "method_atom_energies.json").write_text(json.dumps(method_atom_info, indent=2), encoding="utf-8")
        if verbose:
            print(f"Method atom energies calculated for: {symbols}")
    else:
        method_atom_energies = {}
        if verbose:
            print("Method atom energies skipped: using fitted composition offsets.")

    results = []
    max_workers = max(1, min(HPC.n_workers, len(coord_records)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calculate_one_structure, record, param_toml_path): record
            for record in coord_records
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            if row["status"] == "ok" and verbose:
                print(
                    f"{row['frame_id']}: E_ref={row['ref_energy_ha']:.10f} Ha, "
                    f"E_tblite={row['method_energy_ha']:.10f} Ha, "
                    f"F_shape={row['method_forces_ha_bohr'].shape}, Xi_shape={row['method_virial_ha'].shape}"
                )
            elif row["status"] != "ok":
                print(f"WARNING: {row['frame_id']} failed: {row['reason']}")

    results = sorted(results, key=lambda row: row["frame_index"])
    if energy_mode == "isolated_atoms":
        for row in results:
            if row["status"] == "ok":
                row["ref_ae_ha"] = atomization_energy(row["ref_energy_ha"], row["counts"], ref_atom_energies)
                row["method_ae_ha"] = atomization_energy(row["method_energy_ha"], row["counts"], method_atom_energies)
    return results


def apply_energy_quantities(results, write_metadata=True):
    successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        return {"energy_mode": energy_mode}

    if energy_mode == "fitted_offsets":
        fit_records = [records_by_id[row["frame_id"]] for row in successful]
        ref_totals = np.asarray([row["ref_energy_ha"] for row in successful], dtype=float)
        method_totals = np.asarray([row["method_energy_ha"] for row in successful], dtype=float)
        ref_fit = fit_composition_offsets(ref_totals, fit_records, elements=fit_elements)
        method_fit = fit_composition_offsets(method_totals, fit_records, elements=fit_elements)

        for row, ref_residual, method_residual in zip(successful, ref_fit["residuals"], method_fit["residuals"]):
            row["ref_energy_residual_ha"] = float(ref_residual)
            row["method_energy_residual_ha"] = float(method_residual)

        metadata = {
            "energy_mode": energy_mode,
            "fit_elements": list(fit_elements),
            "reference_offsets_ha": ref_fit["offsets"],
            "method_offsets_ha": method_fit["offsets"],
            "reference_fit_rank": ref_fit["rank"],
            "method_fit_rank": method_fit["rank"],
            "fit_rank": {"reference": ref_fit["rank"], "method": method_fit["rank"]},
            "reference_fit_singular_values": ref_fit["singular_values"],
            "method_fit_singular_values": method_fit["singular_values"],
            "fit_singular_values": {
                "reference": ref_fit["singular_values"],
                "method": method_fit["singular_values"],
            },
        }
        if write_metadata:
            (RUN_OUTPUT_DIR / "energy_offsets.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    if energy_mode == "isolated_atoms":
        for row in successful:
            row["ref_energy_residual_ha"] = float(row["ref_ae_ha"])
            row["method_energy_residual_ha"] = float(row["method_ae_ha"])
        return {
            "energy_mode": energy_mode,
            "fit_elements": [],
            "reference_offsets_ha": {},
            "method_offsets_ha": {},
            "reference_fit_rank": None,
            "method_fit_rank": None,
            "reference_fit_singular_values": [],
            "method_fit_singular_values": [],
        }

    raise ValueError(f"Unknown energy_mode: {energy_mode}")


def compute_offsets_for_vector(vector):
    """Re-run tblite for one parameter vector and return fitted composition offsets.

    Returns the method_offsets_ha dict {element: float_ha} from
    apply_energy_quantities, or an empty dict if calculation fails or
    energy_mode is not fitted_offsets.
    """
    results = calculate(vector, coord_records=records, verbose=False)
    successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        return {}
    metadata = apply_energy_quantities(results, write_metadata=False)
    return metadata.get("method_offsets_ha", {})


# ----------------------- output helpers ----------------------- #

def write_results_csv(results):
    fields = [
        "frame_id",
        "frame_index",
        "status",
        "reason",
        "n_atoms",
        "counts",
        "ref_energy_ha",
        "method_energy_ha",
        "ref_energy_residual_ha",
        "method_energy_residual_ha",
        "energy_residual_error_ha",
        "ref_ae_ha",
        "method_ae_ha",
        "ae_error_ha",
        "force_rmse_ha_bohr",
        "virial_rmse_ha",
        "poscar_path",
        "tblite_out_path",
        "tblite_tag_path",
    ]
    with (RUN_OUTPUT_DIR / "per_structure_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            out = {key: row.get(key, "") for key in fields}
            out["counts"] = json.dumps(row.get("counts", {}), sort_keys=True)
            if row["status"] == "ok":
                if "method_energy_residual_ha" in row and "ref_energy_residual_ha" in row:
                    out["energy_residual_error_ha"] = row["method_energy_residual_ha"] - row["ref_energy_residual_ha"]
                if "method_ae_ha" in row and "ref_ae_ha" in row:
                    out["ae_error_ha"] = row["method_ae_ha"] - row["ref_ae_ha"]
                out["force_rmse_ha_bohr"] = float(np.sqrt(np.mean((row["method_forces_ha_bohr"] - row["ref_forces_ha_bohr"]) ** 2)))
                out["virial_rmse_ha"] = float(np.sqrt(np.mean((row["method_virial_ha"] - row["ref_virial_ha"]) ** 2)))
            writer.writerow(out)




def make_parity_plots(results):
    ok = [row for row in results if row["status"] == "ok"]
    if not ok:
        return
    ha_to_kcal = 627.5094740631

    parity_plot(
        [row["ref_energy_residual_ha"] * ha_to_kcal for row in ok],
        [row["method_energy_residual_ha"] * ha_to_kcal for row in ok],
        "Reference energy residual (kcal/mol)",
        "GFN2-xTB energy residual (kcal/mol)",
        "Energy residual parity",
        RUN_OUTPUT_DIR / "parity_energy_residual_kcalmol.png",
    )
    if energy_mode == "isolated_atoms":
        parity_plot(
            [row["ref_ae_ha"] * ha_to_kcal for row in ok],
            [row["method_ae_ha"] * ha_to_kcal for row in ok],
            "Reference AE (kcal/mol)",
            "GFN2-xTB AE (kcal/mol)",
            "Atomization energy parity",
            RUN_OUTPUT_DIR / "parity_atomization_energy_kcalmol.png",
        )
    force_component_parity_plot(
        [row["ref_forces_ha_bohr"] for row in ok],
        [row["method_forces_ha_bohr"] for row in ok],
        RUN_OUTPUT_DIR / "parity_forces_mevA.png",
    )
    parity_plot(
        np.concatenate([row["ref_virial_ha"].ravel() for row in ok]),
        np.concatenate([row["method_virial_ha"].ravel() for row in ok]),
        "Reference virial (Ha)",
        "GFN2-xTB virial (Ha)",
        "Virial component parity",
        RUN_OUTPUT_DIR / "parity_virials_Ha.png",
    )


# ----------------------- loss definition ----------------------- #

def loss(vector, coord_records=records, wE=wE, wF=wF, wXi=wXi, penalty_value=penalty_value, eps=eps, return_components=False, print_components=False, write_outputs=True, verbose=True):
    param_reg = param_regularization(vector)
    penalty_components = {
        "E": penalty_value,
        "F": penalty_value,
        "Xi": penalty_value,
        "Reg": float(param_reg),
    }
    results = calculate(vector, coord_records, verbose=verbose)
    successful = [row for row in results if row["status"] == "ok"]

    if not successful:
        return (penalty_value, penalty_components) if return_components else penalty_value

    energy_metadata = apply_energy_quantities(results, write_metadata=write_outputs)
    if write_outputs:
        for row in successful:
            save_property_npz(row)

    Eres_ref = np.asarray([row["ref_energy_residual_ha"] for row in successful], dtype=float)
    Eres_pred = np.asarray([row["method_energy_residual_ha"] for row in successful], dtype=float)
    n_atoms = np.asarray([row["n_atoms"] for row in successful], dtype=float)

    dE = Eres_pred - Eres_ref
    wE_i = 1.0 / n_atoms
    sum_wE = np.sum(wE_i)
    mean_dE = np.sum(wE_i * dE) / sum_wE
    mean_Eref = np.sum(wE_i * Eres_ref) / sum_wE
    E_numerator = np.sum(wE_i * (dE - mean_dE) ** 2)
    E_denominator = np.sum(wE_i * (Eres_ref - mean_Eref) ** 2)

    F_numerator = 0.0
    F_denominator = 0.0
    Xi_numerator = 0.0
    Xi_denominator = 0.0

    for row in successful:
        Fref = np.asarray(row["ref_forces_ha_bohr"], dtype=float)
        Fpred = np.asarray(row["method_forces_ha_bohr"], dtype=float)
        if Fref.shape != Fpred.shape or Fref.ndim != 2 or Fref.shape[1] != 3:
            return (penalty_value, penalty_components) if return_components else penalty_value
        Ni = Fref.shape[0]
        dF = Fpred - Fref
        wF_i = 1.0 / (3.0 * Ni)
        F_numerator += wF_i * float(np.sum(dF * dF))
        F_denominator += wF_i * float(np.sum(Fref * Fref))

        Xref = np.asarray(row["ref_virial_ha"], dtype=float)
        Xpred = np.asarray(row["method_virial_ha"], dtype=float)
        if Xref.shape != (3, 3) or Xpred.shape != (3, 3):
            return (penalty_value, penalty_components) if return_components else penalty_value
        dXi = Xpred - Xref
        wXi_i = 1.0 / (9.0 * Ni)
        Xi_numerator += wXi_i * float(np.sum(dXi * dXi))
        Xi_denominator += wXi_i * float(np.sum(Xref * Xref))

    if not np.isfinite(E_denominator) or E_denominator < eps:
        E_term = penalty_value
    else:
        E_term = E_numerator / (E_denominator + eps)
    if not np.isfinite(F_denominator) or F_denominator < eps:
        F_term = penalty_value
    else:
        F_term = F_numerator / (F_denominator + eps)
    if not np.isfinite(Xi_denominator) or Xi_denominator < eps:
        Xi_term = penalty_value
    else:
        Xi_term = Xi_numerator / (Xi_denominator + eps)

    L = wE * E_term + wF * F_term + wXi * Xi_term
    if not np.isfinite(L):
        L = penalty_value

    failed = [{"frame_id": row["frame_id"], "reason": row["reason"]} for row in results if row["status"] != "ok"]
    if write_outputs:
        write_results_csv(results)
        make_parity_plots(results)
        (RUN_OUTPUT_DIR / "failed_structures.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")

    summary = {
        "n_structures_found": len(records_found),
        "n_structures_selected": len(coord_records),
        "split_dataset": split_dataset,
        "train_frac": train_frac,
        "split_seed": split_seed,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "failed_structures": failed,
        "loss_E": float(E_term),
        "loss_F": float(F_term),
        "loss_virial": float(Xi_term),
        "param_reg": float(param_reg),
        "total_loss": float(L),
        "weights": {"E": wE, "F": wF, "Xi": wXi},
        "internal_units": {"energy": "Ha", "energy_residual": "Ha", "forces": "Ha/Bohr", "virial": "Ha"},
        **energy_metadata,
    }
    if write_outputs:
        (RUN_OUTPUT_DIR / "loss_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if print_components:
        print(f"Loss: {L:.10f} (dEres: {E_term:.10f}, dF: {F_term:.10f}, dXi: {Xi_term:.10f}, param_reg: {param_reg:.10f})")
        print(json.dumps(summary, indent=2))

    components = {"E": float(E_term), "F": float(F_term), "Xi": float(Xi_term), "Reg": float(param_reg)}
    return (float(L), components) if return_components else float(L)


# ----------------------- optional diagnostic sweep run ----------------------- #

if calculate_L:
    print("Running test loss calculation...")
    start_time = time.time()
    L = loss(default_gfn2_vector, print_components=True)
    end_time = time.time()
    sweep_time = end_time - start_time
    print(f"Sweep time: {sweep_time:.2f} seconds")
    print(L)


# ----------------------- NSGA-III optimization ----------------------- #

last_loss_x = None
last_loss_result = None


def evaluate_components_for_pymoo(vector, coord_records=records, penalty_value=penalty_value, eps=eps):
    """
    Call loss(..., return_components=True) once and return scalar diagnostics.
    Dead points are marked infeasible and dominated in objective space.
    """
    global last_loss_x, last_loss_result

    x = np.asarray(vector, dtype=float)
    if last_loss_x is not None and np.array_equal(x, last_loss_x):
        return last_loss_result

    L, components = loss(
        x,
        coord_records=coord_records,
        penalty_value=penalty_value,
        eps=eps,
        return_components=True,
        print_components=False,
        write_outputs=False,
        verbose=False,
    )

    E_term = float(components["E"])
    F_term = float(components["F"])
    Xi_term = float(components["Xi"])
    param_reg = float(components["Reg"])
    vals = np.array([L, E_term, F_term, Xi_term, param_reg], dtype=float)

    is_dead = (
        (not np.all(np.isfinite(vals)))
        or (L >= penalty_value)
        or (E_term >= penalty_value)
        or (F_term >= penalty_value)
        or (Xi_term >= penalty_value)
    )
    if is_dead:
        L = penalty_value
        E_term = penalty_value
        F_term = penalty_value
        Xi_term = penalty_value
        param_reg = penalty_value

    result = (
        float(L),
        float(E_term),
        float(F_term),
        float(Xi_term),
        float(param_reg),
        bool(is_dead),
    )
    last_loss_x = x.copy()
    last_loss_result = result
    return result


class GFN2PBCNSGA3Problem(ElementwiseProblem):
    def __init__(
        self,
        lower,
        upper,
        coord_records=records,
        penalty_value=penalty_value,
        eps=eps,
        use_param_regularization=use_param_regularization,
        objectives=objectives,
    ):
        super().__init__(
            n_var=len(lower),
            n_obj=len(objective_names),
            n_ieq_constr=1,
            xl=np.asarray(lower, dtype=float),
            xu=np.asarray(upper, dtype=float),
        )
        self.coord_records = coord_records
        self.penalty_value = penalty_value
        self.eps = eps
        self.use_param_regularization = use_param_regularization
        self.objectives = objectives

    def _evaluate(self, x, out, *args, **kwargs):
        L, E_term, F_term, Xi_term, param_reg, is_dead = evaluate_components_for_pymoo(
            x,
            coord_records=self.coord_records,
            penalty_value=self.penalty_value,
            eps=self.eps,
        )

        component_map = {"E": E_term, "F": F_term, "Xi": Xi_term}
        obj_vals = [component_map[o] for o in self.objectives]
        if self.use_param_regularization:
            obj_vals.append(param_reg)
        out["F"] = np.array(obj_vals, dtype=float)

        out["G"] = np.array([1.0 if is_dead else -1.0], dtype=float)
        out["L"] = float(L)
        out["Reg"] = float(param_reg)




def feasible_nondominated_indices(F, CV=None):
    F = np.asarray(F, dtype=float)
    finite = np.all(np.isfinite(F), axis=1)
    if CV is None:
        feasible = np.ones(F.shape[0], dtype=bool)
    else:
        CV = np.asarray(CV, dtype=float)
        if CV.ndim == 1:
            feasible = CV <= 0.0
        else:
            feasible = np.all(CV <= 0.0, axis=1)
    candidate_idx = np.where(finite & feasible)[0]
    if candidate_idx.size == 0:
        return candidate_idx
    front_local = nondominated_indices(F[candidate_idx])
    return candidate_idx[front_local]


def write_population_csv(path, X, F, CV=None):
    fields = list(objective_names) + [f"param_{name}" for name in names]
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
    def notify(self, algorithm):
        generation = int(getattr(algorithm, "n_gen", 0))
        opt = algorithm.opt
        if opt is None or len(opt) == 0:
            return
        X_front = opt.get("X")
        F_front = opt.get("F")
        CV_front = opt.get("CV")
        if X_front is None or F_front is None:
            return
        save_generation_front(generation, X_front, F_front, CV=CV_front)
        print(f"Saved generation {generation} feasible non-dominated front with {len(X_front)} points.")


if run_optimization:
    number_of_objectives = len(objective_names)
    algo_name = "nsga2" if number_of_objectives == 2 else "nsga3"

    if algo_name == "nsga3":
        ref_dirs = get_reference_directions("energy", number_of_objectives, direction_number, seed=nsga_seed)
        pop_size = len(ref_dirs)
    else:
        pop_size = direction_number  # no ref_dirs needed for NSGA-II

    print(f"Algorithm: {algo_name.upper()}")
    print(f"Number of objectives: {number_of_objectives}")
    print(f"Population size: {pop_size}")
    print(f"Number of generations: {n_gen}")

    X_init = generate_sobol_initial_population(bounds, pop_size, default_gfn2_vector, seed=sobol_seed, scramble=True)

    problem = GFN2PBCNSGA3Problem(
        lower=lower,
        upper=upper,
        coord_records=records,
        penalty_value=penalty_value,
        eps=eps,
        use_param_regularization=use_param_regularization,
        objectives=objectives,
    )

    if algo_name == "nsga3":
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=X_init)
    else:
        algorithm = NSGA2(pop_size=pop_size, sampling=X_init)

    termination = get_termination("n_gen", n_gen)

    start_time = time.time()
    result = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=nsga_seed,
        save_history=True,
        verbose=True,
        callback=GenerationFrontCallback(),
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    nsga_summary = {
        "elapsed_time_seconds": float(elapsed_time),
        "number_of_objectives": number_of_objectives,
        "objective_names": objective_names,
        "population_size": int(pop_size),
        "n_gen": int(n_gen),
        "direction_number": int(direction_number),
        "use_param_regularization": use_param_regularization,
    }

    if result.X is None:
        print("No feasible solutions found.")
        nsga_summary["n_pareto"] = 0
    else:
        X_pareto = result.X
        F_pareto = result.F
        print("# of found Pareto solutions:", X_pareto.shape[0])

        # compute composition offsets for every Pareto solution
        print("Computing train offsets for each Pareto solution...")
        pareto_offsets = []
        for i, x in enumerate(X_pareto):
            print(f"  Offsets {i + 1}/{len(X_pareto)}...")
            pareto_offsets.append(compute_offsets_for_vector(x))
        offset_elements = sorted({k for d in pareto_offsets for k in d})
        print(f"  Done. Offset elements found: {offset_elements}")

        obj_tag = "".join(objective_names)  # e.g. "EF", "EFXi", "EFReg", "EFXiReg"
        results_path = RUN_OUTPUT_DIR / f"{algo_name}_results_pbc_{obj_tag}_{RUN_ID}.npz"
        csv_path = RUN_OUTPUT_DIR / f"{algo_name}_results_pbc_{obj_tag}_{RUN_ID}.csv"
        np.savez_compressed(
            results_path,
            X_pareto=X_pareto,
            F_pareto=F_pareto,
            parameter_names=np.asarray(names),
            objective_names=np.asarray(objective_names),
            bounds=np.asarray(bounds, dtype=float),
            default_gfn2_vector=np.asarray(default_gfn2_vector, dtype=float),
            **{
                f"offset_{el}_ha": np.asarray(
                    [d.get(el, np.nan) for d in pareto_offsets], dtype=float
                )
                for el in offset_elements
            },
        )
        write_population_csv(csv_path, X_pareto, F_pareto, CV=None)

        offsets_csv_path = RUN_OUTPUT_DIR / f"{algo_name}_pareto_train_offsets_{RUN_ID}.csv"
        offset_fields = (
            ["solution_index"]
            + list(objective_names)
            + [f"param_{n}" for n in names]
            + [f"offset_{el}_ha" for el in offset_elements]
        )
        with offsets_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=offset_fields)
            writer.writeheader()
            for i, (x, f, d) in enumerate(zip(X_pareto, F_pareto, pareto_offsets)):
                row = {"solution_index": i}
                for j, oname in enumerate(objective_names):
                    row[oname] = float(f[j])
                for j, pname in enumerate(names):
                    row[f"param_{pname}"] = float(x[j])
                for el in offset_elements:
                    row[f"offset_{el}_ha"] = float(d.get(el, float("nan")))
                writer.writerow(row)
        print(f"Train offsets saved to {offsets_csv_path}.")

        nsga_summary["n_pareto"] = int(X_pareto.shape[0])
        nsga_summary["results_npz"] = str(results_path)
        nsga_summary["results_csv"] = str(csv_path)
        nsga_summary["pareto_train_offsets_csv"] = str(offsets_csv_path)
        print(f"Results saved at {results_path}.")
        print("Printing all results:")
        print(X_pareto)
        print(F_pareto)

    run_config["nsga_summary"] = nsga_summary
    (RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (RUN_OUTPUT_DIR / "nsga3_summary.json").write_text(json.dumps(nsga_summary, indent=2), encoding="utf-8")
