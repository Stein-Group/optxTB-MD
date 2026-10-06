"""
PBC water FIM analysis around the default GFN2-xTB parameter point, with a
selectable set of FIM objectives.

This is the same analysis as pbc_fim_water.py, generalized so the residual
vector fed into the Jacobian/FIM can be restricted to a subset of {E, F, Xi}
via a single keyword:

  fim_objectives = ("E", "F", "Xi")   # full FIM (default, matches pbc_fim_water.py)
  fim_objectives = ("E", "F")         # drop virials from the FIM
  fim_objectives = ("F",)             # forces-only FIM
  ...

The diagnostic scalar loss (printed when calculate_L=True, and used in the
directional scans) always reports all three components (E, F, Xi) regardless
of fim_objectives -- the keyword only controls which residual blocks are
included in the Jacobian/FIM itself.

Usage:

  python -u optimization/T_pbc/pbc_fim_water_select_objectives.py --run-id pbc_fim_obj_smoke

Submit on the cluster with:

  sbatch optimization/T_pbc/run_pbc_fim_water_select_objectives.slurm
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.environ.get('USER', 'user')}"))

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import ensure_param_file
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
    _vector_cache_key,
    compute_fim_from_jacobian,
    eigh_sorted,
    fim_condition_info,
    summarize_eigenvectors,
    write_csv_rows,
    plot_spectrum,
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
        return f"pbc_fim_water_obj_{slurm_job_id}_{stamp}"
    return f"pbc_fim_water_obj_local_{stamp}"


def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Run PBC water FIM analysis (selectable objectives) safely on HPC.")
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
POSCAR_CACHE_DIR = PROJECT_ROOT_PATH / "structures/periodic_data/poscar_cache"

WORKDIR.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROPERTIES_DIR.mkdir(parents=True, exist_ok=True)

print(f"RUN_ID: {RUN_ID}")
print(f"Scratch workdir: {WORKDIR}")
print(f"Persistent output dir: {RUN_OUTPUT_DIR}")
print(f"Python workers: {HPC.n_workers}")


# ----------------------- scientific settings ----------------------- #

dataset_path = PROJECT_ROOT_PATH / "structures/periodic_data/dataset"
isolated_atoms_path = PROJECT_ROOT_PATH / "structures/periodic_data/isolated_atoms"
base_param_file = PROJECT_ROOT_PATH / "gfn2-xtb.toml"

system_filter = "water"  # switch to "all" later after Na reference handling is ready
MAX_STRUCTURES_FOR_TEST = None
split_dataset = False
train_frac = 0.70
split_seed = 20260512
energy_mode = "fitted_offsets"  # "fitted_offsets" or "isolated_atoms"
fit_elements = ("H", "O", "Na", "Cl")

# Keyword for FIM objective choice: any non-empty subset of {"E", "F", "Xi"}.
# Only these residual blocks are concatenated into the FIM Jacobian; the
# diagnostic scalar loss below always reports all three regardless.
fim_objectives = ("E", "F", "Xi")
_valid_objectives = {"E", "F", "Xi"}
if not fim_objectives or not set(fim_objectives).issubset(_valid_objectives):
    raise ValueError(f"fim_objectives must be a non-empty subset of {_valid_objectives}; got {fim_objectives}")
fim_objectives = tuple(o for o in ("E", "F", "Xi") if o in fim_objectives)  # canonical order

wE = 1.0
wF = 1.0
wXi = 1.0
eps = 1e-12
penalty_value = 1e6

tblite_iterations = 2000
tblite_acc = 0.0001
tblite_threads = int(os.environ.get("TBLITE_THREADS", "1"))

calculate_L = True
calculate_FIM = True
finite_difference_h = 1e-2
finite_difference_method = "central"
run_directional_scans = True
scan_amplitudes = np.linspace(-0.5, 0.5, 9)
scan_mode_ids = "auto"  # "auto" means stiff, middle, and sloppiest total modes

print(f"Dataset path: {dataset_path}")
print(f"System filter: {system_filter}")
print(f"MAX_STRUCTURES_FOR_TEST: {MAX_STRUCTURES_FOR_TEST}")
print(f"Split dataset: {split_dataset}")
print(f"Train fraction: {train_frac}, split seed: {split_seed}")
print(f"Energy mode: {energy_mode}")
print(f"Fit elements: {fit_elements}")
print(f"FIM objectives: {fim_objectives}")
print(f"Loss component weights: wE={wE}, wF={wF}, wXi={wXi}")
print(f"tblite threads per worker: {tblite_threads}")
print(f"calculate_L: {calculate_L}, calculate_FIM: {calculate_FIM}")


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

fim_param_changes = {
    "kpair": {
        "H-H": param_changes["kpair"]["H-H"],
        "O-O": param_changes["kpair"]["O-O"],
        "O-H": param_changes["kpair"]["O-H"],
    },
    "H": param_changes["H"],
    "O": param_changes["O"],
}



names, bounds, default_gfn2_vector, structure = generate_parameter_info(
    fim_param_changes,
)
param_scale = 0.5 * (np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0])
print("Parameter names:")
print(names)
print("Bounds:")
print(bounds)
print(f"Number of included parameters {len(names)}")


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
    "fim_objectives": list(fim_objectives),
    "weights": {"E": wE, "F": wF, "Xi": wXi},
    "eps": eps,
    "calculate_L": calculate_L,
    "calculate_FIM": calculate_FIM,
    "finite_difference_h": finite_difference_h,
    "finite_difference_method": finite_difference_method,
    "fim_parameter_names": names,
    "fim_bounds": bounds,
    "fim_param_scale": param_scale.tolist(),
    "fim_varied_parameter_scope": "H/O element parameters plus H-H, O-O, O-H kpair terms",
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


def calculate(vector, coord_records=records):
    param_dict = vector_to_param_changes(vector, structure)
    param_toml_path = ensure_param_file(param_dict, param_file=str(base_param_file), DIR=WORKDIR)
    run_config["param_toml_path"] = str(param_toml_path)
    (RUN_OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
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
        print(f"Method atom energies calculated for: {symbols}")
    else:
        method_atom_energies = {}
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
            if row["status"] == "ok":
                print(
                    f"{row['frame_id']}: E_ref={row['ref_energy_ha']:.10f} Ha, "
                    f"E_tblite={row['method_energy_ha']:.10f} Ha, "
                    f"F_shape={row['method_forces_ha_bohr'].shape}, Xi_shape={row['method_virial_ha'].shape}"
                )
            else:
                print(f"WARNING: {row['frame_id']} failed: {row['reason']}")

    results = sorted(results, key=lambda row: row["frame_index"])
    if energy_mode == "isolated_atoms":
        for row in results:
            if row["status"] == "ok":
                row["ref_ae_ha"] = atomization_energy(row["ref_energy_ha"], row["counts"], ref_atom_energies)
                row["method_ae_ha"] = atomization_energy(row["method_energy_ha"], row["counts"], method_atom_energies)
    return results


def apply_energy_quantities(results):
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


# ----------------------- loss definition (always full E/F/Xi diagnostic) ----------------------- #

def loss(vector, coord_records=records, wE=wE, wF=wF, wXi=wXi, penalty_value=penalty_value, eps=eps, return_components=False, print_components=False):
    results = calculate(vector, coord_records)
    successful = [row for row in results if row["status"] == "ok"]

    if not successful:
        components = {"E": penalty_value, "F": penalty_value, "Xi": penalty_value}
        return (penalty_value, components) if return_components else penalty_value

    energy_metadata = apply_energy_quantities(results)
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
            components = {"E": penalty_value, "F": penalty_value, "Xi": penalty_value}
            return (penalty_value, components) if return_components else penalty_value
        Ni = Fref.shape[0]
        dF = Fpred - Fref
        wF_i = 1.0 / (3.0 * Ni)
        F_numerator += wF_i * float(np.sum(dF * dF))
        F_denominator += wF_i * float(np.sum(Fref * Fref))

        Xref = np.asarray(row["ref_virial_ha"], dtype=float)
        Xpred = np.asarray(row["method_virial_ha"], dtype=float)
        if Xref.shape != (3, 3) or Xpred.shape != (3, 3):
            components = {"E": penalty_value, "F": penalty_value, "Xi": penalty_value}
            return (penalty_value, components) if return_components else penalty_value
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
        "total_loss": float(L),
        "weights": {"E": wE, "F": wF, "Xi": wXi},
        "internal_units": {"energy": "Ha", "energy_residual": "Ha", "forces": "Ha/Bohr", "virial": "Ha"},
        **energy_metadata,
    }
    (RUN_OUTPUT_DIR / "loss_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if print_components:
        print(f"Loss: {L:.10f} (dEres: {E_term:.10f}, dF: {F_term:.10f}, dXi: {Xi_term:.10f})")
        print(json.dumps(summary, indent=2))

    components = {"E": float(E_term), "F": float(F_term), "Xi": float(Xi_term)}
    return (float(L), components) if return_components else float(L)


# ----------------------- FIM / Gauss-Newton helpers (objective-selectable) ----------------------- #

@dataclass
class ResidualNormCache:
    denominators: dict = field(default_factory=dict)  # objective -> normalization denominator
    wE_i: np.ndarray = None
    frame_ids: tuple = ()
    sizes: dict = field(default_factory=dict)  # objective -> residual block size



_calculate_cache = {}


def calculate_cached(vector, coord_records=records, use_cache=True):
    key = (_vector_cache_key(vector), tuple(record.frame_id for record in coord_records))
    if use_cache and key in _calculate_cache:
        return _calculate_cache[key]
    out = calculate(vector, coord_records)
    if use_cache:
        _calculate_cache[key] = out
    return out


def successful_rows_for_vector(vector, coord_records=records, use_cache=True):
    results = calculate_cached(vector, coord_records=coord_records, use_cache=use_cache)
    successful = [row for row in results if row["status"] == "ok"]
    failed = [{"frame_id": row["frame_id"], "reason": row["reason"]} for row in results if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"FIM requires identical successful structures; failures: {failed}")
    apply_energy_quantities(results)
    return successful


def make_residual_norm_cache(successful, fim_objectives=fim_objectives, eps=eps):
    n_atoms = np.asarray([row["n_atoms"] for row in successful], dtype=float)
    wE_i = 1.0 / n_atoms

    denominators = {}
    sizes = {}

    if "E" in fim_objectives:
        Eres_ref = np.asarray([row["ref_energy_residual_ha"] for row in successful], dtype=float)
        mean_Eref = np.sum(wE_i * Eres_ref) / np.sum(wE_i)
        denominators["E"] = float(np.sum(wE_i * (Eres_ref - mean_Eref) ** 2))
        sizes["E"] = len(successful)

    if "F" in fim_objectives:
        F_denominator = 0.0
        nF = 0
        for row in successful:
            Fref = np.asarray(row["ref_forces_ha_bohr"], dtype=float)
            Ni = row["n_atoms"]
            F_denominator += (1.0 / (3.0 * Ni)) * float(np.sum(Fref * Fref))
            nF += Fref.size
        denominators["F"] = float(F_denominator)
        sizes["F"] = int(nF)

    if "Xi" in fim_objectives:
        Xi_denominator = 0.0
        nXi = 0
        for row in successful:
            Xref = np.asarray(row["ref_virial_ha"], dtype=float)
            Ni = row["n_atoms"]
            Xi_denominator += (1.0 / (9.0 * Ni)) * float(np.sum(Xref * Xref))
            nXi += Xref.size
        denominators["Xi"] = float(Xi_denominator)
        sizes["Xi"] = int(nXi)

    for obj, denom in denominators.items():
        if not np.isfinite(denom) or denom < eps:
            raise ValueError(f"Bad {obj}_denominator: {denom}")

    return ResidualNormCache(
        denominators=denominators,
        wE_i=wE_i,
        frame_ids=tuple(row["frame_id"] for row in successful),
        sizes=sizes,
    )


def residual_vector(
    vector,
    coord_records=records,
    wE=wE,
    wF=wF,
    wXi=wXi,
    eps=eps,
    norm_cache=None,
    use_cache=True,
    return_slices=False,
    fim_objectives=fim_objectives,
):
    """
    Build r such that dot(r, r) reproduces the scalar PBC loss restricted to the
    objectives in fim_objectives (a subset of {E, F, Xi}).
    """
    weights = {"E": wE, "F": wF, "Xi": wXi}
    successful = successful_rows_for_vector(vector, coord_records=coord_records, use_cache=use_cache)
    if norm_cache is None:
        norm_cache = make_residual_norm_cache(successful, fim_objectives=fim_objectives, eps=eps)

    frame_ids = tuple(row["frame_id"] for row in successful)
    if frame_ids != norm_cache.frame_ids:
        raise RuntimeError("The successful structure set changed during FIM finite differences.")

    parts = {}

    if "E" in fim_objectives:
        Eres_ref = np.asarray([row["ref_energy_residual_ha"] for row in successful], dtype=float)
        Eres_pred = np.asarray([row["method_energy_residual_ha"] for row in successful], dtype=float)
        dE = Eres_pred - Eres_ref
        mean_dE = np.sum(norm_cache.wE_i * dE) / np.sum(norm_cache.wE_i)
        parts["E"] = np.sqrt(weights["E"] * norm_cache.wE_i / (norm_cache.denominators["E"] + eps)) * (dE - mean_dE)

    if "F" in fim_objectives:
        rF_parts = []
        for row in successful:
            Fref = np.asarray(row["ref_forces_ha_bohr"], dtype=float)
            Fpred = np.asarray(row["method_forces_ha_bohr"], dtype=float)
            if Fref.shape != Fpred.shape or Fref.ndim != 2 or Fref.shape[1] != 3:
                raise RuntimeError(f"Bad force shapes for {row['frame_id']}: {Fref.shape} vs {Fpred.shape}")
            Ni = row["n_atoms"]
            rF_parts.append(
                (np.sqrt(weights["F"] * (1.0 / (3.0 * Ni)) / (norm_cache.denominators["F"] + eps)) * (Fpred - Fref)).reshape(-1)
            )
        parts["F"] = np.concatenate(rF_parts, axis=0)

    if "Xi" in fim_objectives:
        rXi_parts = []
        for row in successful:
            Xref = np.asarray(row["ref_virial_ha"], dtype=float)
            Xpred = np.asarray(row["method_virial_ha"], dtype=float)
            if Xref.shape != (3, 3) or Xpred.shape != (3, 3):
                raise RuntimeError(f"Bad virial shapes for {row['frame_id']}: {Xref.shape} vs {Xpred.shape}")
            Ni = row["n_atoms"]
            rXi_parts.append(
                (np.sqrt(weights["Xi"] * (1.0 / (9.0 * Ni)) / (norm_cache.denominators["Xi"] + eps)) * (Xpred - Xref)).reshape(-1)
            )
        parts["Xi"] = np.concatenate(rXi_parts, axis=0)

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
            "Xi": float(comps["Xi"]),
            "hit_lower": hit_lower,
            "hit_upper": hit_upper,
        })
    return rows


def plot_directional_scan(rows, lambda_k, out_path, title):
    amp = np.asarray([row["amp"] for row in rows], dtype=float)
    L = np.asarray([row["L"] for row in rows], dtype=float)
    E = np.asarray([row["E"] for row in rows], dtype=float)
    F = np.asarray([row["F"] for row in rows], dtype=float)
    Xi = np.asarray([row["Xi"] for row in rows], dtype=float)
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
    axes[1].plot(amp, Xi, marker="o", label="Xi")
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

    baseline_rows = successful_rows_for_vector(theta_ref, use_cache=True)
    norm_cache = make_residual_norm_cache(baseline_rows, fim_objectives=fim_objectives)
    r_ref, slices_ref, parts_ref = residual_vector(
        theta_ref,
        norm_cache=norm_cache,
        use_cache=True,
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
        return residual_vector(
            theta_raw,
            norm_cache=norm_cache,
            use_cache=True,
            return_slices=False,
            fim_objectives=fim_objectives,
        )

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
    spectrum_label = {"E": "energy", "F": "forces", "Xi": "virials"}
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
            ["mode", "lambda_total", "amp", "L", "E", "F", "Xi", "hit_lower", "hit_upper"],
        )

    summary = {
        "run_id": RUN_ID,
        "n_structures_found": len(records_found),
        "n_structures_selected": len(records),
        "n_parameters": len(names),
        "parameter_names": names,
        "parameter_scope": "H/O element parameters plus H-H, O-O, O-H kpair terms",
        "fim_objectives": list(fim_objectives),
        "theta_ref_label": "default_gfn2",
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
    print("Running PBC water FIM analysis...")
    run_fim_analysis()
