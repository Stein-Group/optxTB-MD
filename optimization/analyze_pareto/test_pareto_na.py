"""
Validate selected Na Pareto points against Na+-H2O overbinding probes:

  - Na+-H2O 1:1 probe: a fresh, simple rigid scan of a single water molecule
    approaching Na+ along its lone-pair direction, compared against a
    Q-Chem revPBE0/aug-cc-pVTZ reference scan
    (structures/cluster_data/na_h2o_11_scan). A Morse fit gives a well depth
    D_e (eV) and equilibrium Na-O distance R_e (A).
  - Na+(H2O)4 tetrahedral rigid scan: all four Na-O distances scaled together,
    compared against a Q-Chem revPBE0/aug-cc-pVTZ reference
    (structures/cluster_data/na_h2o4_tet_scan).

POINT_INDICES below is a list; one set of output files is written per index.
The default GFN2-xTB curves (baseline) are computed once and cached as NPZ
files in RUN_OUTPUT_DIR so they are reloaded on subsequent runs with the same
--run-id. Per-point selected curves are cached the same way.

Submit with:

  sbatch optimization/analyze_pareto/submit_test_pareto_na.slurm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.environ.get('USER', 'user')}"))

import matplotlib.pyplot as plt
import numpy as np
import toml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import Bohr, Hartree
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import ensure_param_file, get_base_param_file, get_total_charge, get_number_of_na_cl_waters
from codes.gradients_qchem import calculate_re_f_md_dft
from codes.pbc_data import composition_residuals, ensure_poscar_cache, fit_composition_offsets, load_pbc_dataset
from codes.pbc_tblite import run_tblite_molecule, run_tblite_pbc
from codes.common import (
    init_tblite_env,
    load_fixed_params_from_toml,
    merge_param_changes,
)


# ----------------------- what to test ----------------------- #
calculate_one_water = False  # include the Na+-H2O 1:1 probe panel
calculate_four_waters = True  # include the Na+(H2O)4 tetrahedral rigid scan diagnostic
evaluate_opt_water_default_na = False
if evaluate_opt_water_default_na:
    print('Alarm: calculating H/O pure effect on Na-O...')

include_repulsion = True

#type_data = 'pbc'
#SOURCE_RUN_ID = "nsga3_pbc_na_212897_20260611_101143" # 37
#SOURCE_RUN_ID = "nsga3_pbc_na_213525_20260617_180557" # 3
#SOURCE_RUN_ID = "nsga3_pbc_na_212898_20260611_101624" # 11
#SOURCE_RUN_ID = "nsga3_pbc_na_continue_215356_20260703_092115" # 3 with continuation
#SOURCE_RUN_ID = "nsga3_pbc_na_215557_20260708_143844" # 3 with arep/zeff
#SOURCE_RUN_ID = "nsga3_pbc_na_215845_20260719_031739" # Na-O PBC scan with arep/zeff
#SOURCE_RUN_ID = "nsga3_pbc_na_215846_20260719_154003" # Na-O cluster in a box scan with arep/zeff
#CHARGE_SCHEME = None
# cluster
type_data = 'cluster'
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_constr_215005_20260626_140824" # 246 and IH
#CHARGE_SCHEME = "HirshfeldI"
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_constr_215006_20260626_140853" # 246 and CM5
#CHARGE_SCHEME = "CM5"
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_reg_215029_20260628_120326" # 246 and E,F with Reg
#CHARGE_SCHEME = None
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_reg_215379_20260703_120831" # 246 and IH (with Reg) and Na-O scan
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_constr_215382_20260703_121149" # 246 and IH (with constr) and Na-O scan
#CHARGE_SCHEME = "HirshfeldI"
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_constr_215383_20260703_121432" # 246 and CM5 (with constr) and Na-O scan
#CHARGE_SCHEME = "CM5"
#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_reg_215507_20260707_175950"
#CHARGE_SCHEME = "HirshfeldI"

#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_reg_215508_20260707_180101" # (E, F) with Reg for Na-O scans
#CHARGE_SCHEME = None
#include_repulsion = True

#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_reg_216809_20260805_164035" # (E, F) with Reg for Na-O scans on top of H/O 252 (6 params + O-O scans)
#CHARGE_SCHEME = None

#SOURCE_RUN_ID = "nsga3_t2na_t3_efq_na_reg_216816_20260806_015821" # (E, F) with Reg for Na-O scans on top of H/O 249 (6 params + O-O scans)
#CHARGE_SCHEME = None

#SOURCE_RUN_ID = 'nsga3_t2na_t3_efq_na_reg_216861_20260806_112033' # (E, F, HI) for H/O 252
#CHARGE_SCHEME = "HirshfeldI"

#SOURCE_RUN_ID = 'nsga3_t2na_t3_efq_na_reg_216862_20260806_163711' # (E, F, CM5) for H/O 252
#CHARGE_SCHEME = "CM5"

SOURCE_RUN_ID = 'nsga3_t2na_t3_efq_na_reg_216898_20260806_174242' # (E, F, HI) for H/O 249
CHARGE_SCHEME = "HirshfeldI"

#SOURCE_RUN_ID = 'nsga3_t2na_t3_efq_na_reg_216899_20260809_183219' # (E, F, CM5) for H/O 249
#CHARGE_SCHEME = "CM5"

#POINT_INDICES = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
#POINT_INDICES = [ 4,  8, 12, 14, 17, 19, 36, 45, 48, 50, 56, 61, 64, 70, 72, 78, 87, 89]
#POINT_INDICES = [ 2,  3, 21, 22, 27, 36, 46, 65, 68, 78]
POINT_INDICES = [89]

# If True, additionally evaluate E/F/(Xi, PBC only) parity -- restricted to the Na-related
# subset of the dataset (T2_Na + T3 for cluster; system_filter="nacl" for PBC) -- against the
# *same* train/test split saved by the original Na optimization script, for default GFN2-xTB
# and every selected point, under RUN_OUTPUT_DIR/parity_plots/. Requires the source run's
# dataset_split.json (written by nsga3_loss_regularized_Na.py / nsga3_pbc_loss_regularized_Na.py).
# If False, the script behaves exactly as before.
PLOT_PARITY = True
PARITY_MAX_STRUCTURES = None  # debug knob: cap #structures (train+test) used for parity; None = full split
PARITY_SPLITS = ("test",)  # which split(s) to actually plot/save: subset of ("train", "test").

if CHARGE_SCHEME:
    print(f"Charge scheme: {CHARGE_SCHEME}")
OUR_LABEL = "Selected Pareto point" + (f" ({CHARGE_SCHEME})" if CHARGE_SCHEME else "")

HEXAMER_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/water_hexamers"
QCHEM_11_REF_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/na_h2o_11_scan"
QCHEM_H2O4_REF_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/na_h2o4_tet_scan"
NA_H2O4_TEMPLATE = PROJECT_ROOT_PATH / "structures/cluster_data/dataset/117_Na_H2O_4.xyz"

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.503
QCHEM_ENERGY_PATTERN = re.compile(r"Total energy in the final basis set =\s*([-+]?\d*\.\d+|\d+)")
SCAN_ETEMP = 1000.0  # K
NEAR_MAX_R_A = 8.0


# ----------------------- HPC run plumbing ----------------------- #


init_tblite_env(omp_threads=1, blas_threads=1, amd_mkl=True)


def _default_scratch_root():
    for path in [Path(p) for p in (os.environ.get("SCRATCH"), "/tmp") if p]:
        if path.exists() and os.access(path, os.W_OK):
            return path
    return Path("/tmp")


DEFAULT_SCRATCH_ROOT = _default_scratch_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_PATH / "optimization/analyze_pareto/pareto_tests"


def _default_run_id():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"test_pareto_na_{slurm_job_id}_{stamp}" if slurm_job_id else f"test_pareto_na_local_{stamp}"


def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Validate Na Pareto points' Na+-H2O overbinding.")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", _default_run_id()))
    parser.add_argument("--scratch-root", default=os.environ.get("SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT)))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    args, _unknown = parser.parse_known_args()
    return args


HPC = parse_hpc_args()
RUN_ID = HPC.run_id
SCRATCH_DIR = Path(HPC.scratch_root) / "optxTB-MD" / RUN_ID
WORKDIR = SCRATCH_DIR / "work"
RUN_OUTPUT_DIR = Path(HPC.output_root) / RUN_ID
MOLECULE_RUNS = WORKDIR / "molecule_tblite_runs"

WORKDIR.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"RUN_ID: {RUN_ID}")
print(f"Scratch workdir: {WORKDIR}")
print(f"Persistent output dir: {RUN_OUTPUT_DIR}")
print(f"Source Pareto run: {SOURCE_RUN_ID}, point indices: {POINT_INDICES}")


# ----------------------- load the Pareto front ----------------------- #


if type_data == 'pbc':
    SOURCE_RUN_DIR = PROJECT_ROOT_PATH / "optimization/T_pbc/hpc_runs" / SOURCE_RUN_ID
elif type_data == 'cluster':
    SOURCE_RUN_DIR = PROJECT_ROOT_PATH / "optimization/T_all_partial_charge_schemes/hpc_runs" / SOURCE_RUN_ID
run_config = json.loads((SOURCE_RUN_DIR / "run_config.json").read_text())
if type_data == 'pbc':
    npz_path = next(SOURCE_RUN_DIR.glob(f"*results_pbc*{SOURCE_RUN_ID}.npz"))
elif type_data == 'cluster':
    npz_path = next(SOURCE_RUN_DIR.glob(f"*results_cluster*{SOURCE_RUN_ID}.npz"))
pareto = np.load(npz_path, allow_pickle=True)
try:
    X_pareto = pareto["X_pareto"]
except:
    X_pareto = pareto["X"]
parameter_names = [str(n) for n in pareto["parameter_names"]]
print(f"Loaded Pareto front: {npz_path.name}")
print(f"Parameter names: {parameter_names}")

base_param_file = Path(run_config.get("base_param_file", str(get_base_param_file())))
if not base_param_file.exists():
    base_param_file = get_base_param_file()

fixed_ho_toml_path = Path(run_config["fixed_ho_toml_path"])
if type_data == 'pbc':
    fixed_element_params = run_config["fixed_element_params"]
    fixed_kpair_keys = run_config["fixed_kpair_keys"]
    print(f"Base param file: {base_param_file}")
    print(f"Fixed H/O parameter file: {fixed_ho_toml_path}")
elif type_data == 'cluster':
    fixed_element_params = {"H": ("zeff", "arep"), "O": ("zeff", "arep")}
    fixed_kpair_keys = ("O-H",)
    print(f"Fixed H/O parameter file: {fixed_ho_toml_path}")
    print("Fixed H/O and water kpair parameters:")
    print(load_fixed_params_from_toml(fixed_ho_toml_path, fixed_element_params, fixed_kpair_keys))

if include_repulsion:
    param_changes = {
        "kpair": {"Na-O": 1.0, "Na-H": 1.0},
        "Na": {"slater": [0.7637870000000000, 0.5735530000000000], "gam": 0.2710560000000000, "zeff": 5.2449170000000001, "arep": 0.5727280000000000},
    }
else:
    param_changes = {
        "kpair": {"Na-O": 1.0, "Na-H": 1.0},
        "Na": {"slater": [0.7637870000000000, 0.5735530000000000], "gam": 0.2710560000000000},
    }


def generate_parameter_info(param_changes, variation=0.30, kernel_abs=0.03, near_one_width=0.20, positivity_eps=1e-12, fallback_zero_abs=0.20, fixed_components=None):
    if fixed_components is None:
        fixed_components = set()

    def _percent_box(val):
        if val == 0:
            return [-fallback_zero_abs, +fallback_zero_abs]
        lo = min(val * (1 - variation), val * (1 + variation))
        hi = max(val * (1 - variation), val * (1 + variation))
        return [lo, hi]

    def _near_one_box():
        return [max(positivity_eps, 1.0 - near_one_width), 1.0 + near_one_width]

    def _make_bound(elem, key, val, idx=None):
        if key == "dkernel" and elem == "Na":
            lo, hi = -abs(kernel_abs), abs(kernel_abs)
        elif key in {"Na-O", "Na-H", "Cl-H", "Cl-O", "H-H", "O-O", "O-H"}:
            lo, hi = _near_one_box()
        elif key == "gam":
            lo, hi = _percent_box(val)
            lo = max(lo, positivity_eps)
        else:
            lo, hi = _percent_box(val)
        if not (hi > lo):
            mid = 0.5 * (lo + hi)
            tiny = max(1e-9, abs(mid) * 1e-12)
            lo, hi = mid - tiny, mid + tiny
        return [lo, hi]

    names, bounds, vector, structure = [], [], [], {}
    for elem, params in param_changes.items():
        structure[elem] = {}
        for key, value in params.items():
            if isinstance(value, list):
                structure[elem][key] = []
                for i, val in enumerate(value):
                    if key == "lgam" and i == 0:
                        structure[elem][key].append("Fixing to 1.0")
                        continue
                    if (elem, key, i) in fixed_components:
                        structure[elem][key].append(("__fixed__", val))
                        continue
                    par_name = f"{elem}_{key}_{i}"
                    names.append(par_name)
                    bounds.append(_make_bound(elem, key, val, idx=i))
                    vector.append(val)
                    structure[elem][key].append(par_name)
            else:
                par_name = f"{elem}_{key}"
                names.append(par_name)
                bounds.append(_make_bound(elem, key, value))
                vector.append(value)
                structure[elem][key] = par_name
    return names, bounds, vector, structure


def print_bound_proximity(param_names, vector, bounds, label, warn_frac=0.05):
    """Print, per parameter, how close the selected value sits to its lower/upper bound."""
    print(f"[{label}] Parameter proximity to bounds:")
    for name, val, (lo, hi) in zip(param_names, vector, bounds):
        rng = hi - lo
        if rng <= 0:
            print(f"    {name}: {val:.6g}  [{lo:.6g}, {hi:.6g}]  (zero-width bound)")
            continue
        frac_lo = (val - lo) / rng
        frac_hi = (hi - val) / rng
        flag = ""
        if frac_lo <= warn_frac:
            flag = "  <-- near LOWER bound"
        elif frac_hi <= warn_frac:
            flag = "  <-- near UPPER bound"
        print(f"    {name}: {val:.6g}  [{lo:.6g}, {hi:.6g}]  "
              f"(from lo: {frac_lo*100:.1f}%, from hi: {frac_hi*100:.1f}%){flag}")



names, _bounds, _default_vector, structure = generate_parameter_info(
    param_changes, fixed_components={("Na", "slater", 1)},
)
assert names == parameter_names, f"Parameter order mismatch: {names} vs {parameter_names}"

# fixed H/O changes are the same for every Pareto point
fixed_changes = load_fixed_params_from_toml(fixed_ho_toml_path, fixed_element_params, fixed_kpair_keys)


# ----------------------- parity plots (optional, PLOT_PARITY) ----------------------- #
#
# Reproduces the E/F/(Xi) parity-plot style built for test_pareto_water.py, restricted to the
# Na-related subset of the dataset (cluster: T2_Na + T3 tiers; PBC: system_filter="nacl"), on
# the *same* train/test split saved by the original Na optimization script (loaded directly from
# dataset_split.json rather than recomputed -- the cluster split is F_term-stratified and not
# worth reproducing independently when the exact frame numbers/ids are already on disk). Virial
# (Xi) parity only applies to PBC data -- clusters have no defined virial/stress tensor.

if PLOT_PARITY:
    PARITY_DIR = RUN_OUTPUT_DIR / "parity_plots"
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    PARITY_WORKERS = int(os.environ.get("PARITY_WORKERS", 32))
    FORCE_HA_BOHR_TO_MEV_A = 1000.0 * HARTREE_TO_EV / 0.529177210903

    dataset_split_path = SOURCE_RUN_DIR / "dataset_split.json"
    if not dataset_split_path.exists():
        raise FileNotFoundError(
            f"PLOT_PARITY=True needs {dataset_split_path} (train/test split saved by the Na "
            "optimization script); this source run predates that file, or was run with "
            "split_dataset=False."
        )
    dataset_split = json.loads(dataset_split_path.read_text())
    CHARGE_SCHEME_RUN = run_config.get("charge_scheme", "N/A")  # exact scheme name/casing used
    # by the source run (matches the dft_multiwfn_chg_revpbe0/<scheme> dir name), NOT necessarily
    # the same casing as the top-level CHARGE_SCHEME label above.

    def directional_deviation(F_ref_list, F_pred_list):
        """[1 - cos(F_ref, F_pred)] * |F_pred|, per atom, flattened over all structures."""
        scalar, magnitude = [], []
        for F_ref, F_pred in zip(F_ref_list, F_pred_list):
            F_ref = np.asarray(F_ref, dtype=float)
            F_pred = np.asarray(F_pred, dtype=float)
            for atom in range(len(F_ref)):
                mag_ref = np.linalg.norm(F_ref[atom])
                mag_pred = np.linalg.norm(F_pred[atom])
                unit_ref = F_ref[atom] / (mag_ref + 1e-12)
                unit_pred = F_pred[atom] / (mag_pred + 1e-12)
                scalar.append(1 - np.dot(unit_ref, unit_pred))
                magnitude.append(mag_pred)
        return np.asarray(scalar) * np.asarray(magnitude)

    def plot_energy_parity(E_ref, E_base, E_sel, title, out_path):
        lo, hi = min(E_ref.min(), E_base.min(), E_sel.min()), max(E_ref.max(), E_base.max(), E_sel.max())
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(E_ref, E_base, alpha=0.5, color="0.5",
                   label=f"default, R$^2$={r2_score(E_ref, E_base):.4f}, RMSE={root_mean_squared_error(E_ref, E_base):.3f} kcal/mol")
        ax.scatter(E_ref, E_sel, alpha=0.5, color="tab:blue",
                   label=f"selected, R$^2$={r2_score(E_ref, E_sel):.4f}, RMSE={root_mean_squared_error(E_ref, E_sel):.3f} kcal/mol")
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel("Reference energy (kcal/mol)")
        ax.set_ylabel("Predicted energy (kcal/mol)")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_force_components(F_ref, F_base, F_sel, title, out_path):
        component_names = ["x", "y", "z"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for comp, ax in enumerate(axes):
            r, b, s = F_ref[:, comp], F_base[:, comp], F_sel[:, comp]
            ax.scatter(r, b, alpha=0.4, s=9, color="0.5",
                       label=f"default, R$^2$={r2_score(r, b):.4f}, RMSE={root_mean_squared_error(r, b):.2f} meV/Å")
            ax.scatter(r, s, alpha=0.4, s=9, color="tab:blue",
                       label=f"selected, R$^2$={r2_score(r, s):.4f}, RMSE={root_mean_squared_error(r, s):.2f} meV/Å")
            lo, hi = min(r.min(), b.min(), s.min()), max(r.max(), b.max(), s.max())
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set_title(f"Component {component_names[comp]}")
            ax.set_xlabel("Reference force (meV/Å)")
            if comp == 0:
                ax.set_ylabel("Predicted force (meV/Å)")
            ax.grid(True)
            ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_directional(dev_base, dev_sel, title, out_path):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for ax, dev, name, color in (
            (axs[0], dev_base, "GFN2-xTB (default)", "tab:orange"),
            (axs[1], dev_sel, "GFN2-xTB (selected)", "tab:blue"),
        ):
            ax.scatter(np.arange(len(dev)), dev, alpha=0.5, color=color,
                       label=f"Mean {dev.mean():.3f}, median {np.median(dev):.3f} meV/Å")
            ax.set_title(f"{name}: Directional Deviation ({title})")
            ax.set_xlabel("Atom index")
            ax.set_ylabel("[1-cos(F_ref,F_pred)] * |F_pred| (meV/Å)", fontsize=9)
            ax.grid()
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_virial_parity(Xi_ref, Xi_base, Xi_sel, title, out_path):
        lo, hi = min(Xi_ref.min(), Xi_base.min(), Xi_sel.min()), max(Xi_ref.max(), Xi_base.max(), Xi_sel.max())
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(Xi_ref, Xi_base, alpha=0.5, color="0.5",
                   label=f"default, R$^2$={r2_score(Xi_ref, Xi_base):.4f}, RMSE={root_mean_squared_error(Xi_ref, Xi_base):.4f} Ha")
        ax.scatter(Xi_ref, Xi_sel, alpha=0.5, color="tab:blue",
                   label=f"selected, R$^2$={r2_score(Xi_ref, Xi_sel):.4f}, RMSE={root_mean_squared_error(Xi_ref, Xi_sel):.4f} Ha")
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel("Reference virial (Ha)")
        ax.set_ylabel("Predicted virial (Ha)")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_charge_parity(Q_ref, Q_base, Q_sel, title, out_path):
        lo, hi = min(Q_ref.min(), Q_base.min(), Q_sel.min()), max(Q_ref.max(), Q_base.max(), Q_sel.max())
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(Q_ref, Q_base, alpha=0.5, color="0.5",
                   label=f"default, R$^2$={r2_score(Q_ref, Q_base):.4f}, RMSE={root_mean_squared_error(Q_ref, Q_base):.4f} e")
        ax.scatter(Q_ref, Q_sel, alpha=0.5, color="tab:blue",
                   label=f"selected, R$^2$={r2_score(Q_ref, Q_sel):.4f}, RMSE={root_mean_squared_error(Q_ref, Q_sel):.4f} e")
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel("Reference partial charge (e)")
        ax.set_ylabel("Predicted partial charge (e)")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def parity_plots_for_point(label, point_dir, E_ref, F_ref, Xi_ref, Q_ref, E_base, F_base, Xi_base, Q_base,
                                E_sel, F_sel, Xi_sel, Q_sel, train_idx, test_idx, has_virial, has_charges, splits):
        """Plot + save (single npz) E/F/(Xi)/(Q) parity for one Pareto point, on the requested splits.

        E_*/F_*/Xi_*/Q_* may be plain lists with None entries outside `splits` (structures that
        were skipped, or predicted energy-only) -- always indexed element-wise, never bulk
        np.asarray()'d, so those None entries are never touched as long as `splits` matches what
        was actually predicted upstream.
        """
        point_dir.mkdir(parents=True, exist_ok=True)
        npz_data = {}
        for split_name, idx in (("train", train_idx), ("test", test_idx)):
            if split_name not in splits or len(idx) == 0:
                continue
            e_ref = np.asarray([E_ref[i] for i in idx]) * HARTREE_TO_KCAL
            e_base = np.asarray([E_base[i] for i in idx]) * HARTREE_TO_KCAL
            e_sel = np.asarray([E_sel[i] for i in idx]) * HARTREE_TO_KCAL
            f_ref = [F_ref[i] * FORCE_HA_BOHR_TO_MEV_A for i in idx]
            f_base = [F_base[i] * FORCE_HA_BOHR_TO_MEV_A for i in idx]
            f_sel = [F_sel[i] * FORCE_HA_BOHR_TO_MEV_A for i in idx]
            f_ref_flat = np.concatenate(f_ref, axis=0)
            f_base_flat = np.concatenate(f_base, axis=0)
            f_sel_flat = np.concatenate(f_sel, axis=0)

            npz_data[f"{split_name}_E_ref_kcalmol"] = e_ref
            npz_data[f"{split_name}_E_baseline_kcalmol"] = e_base
            npz_data[f"{split_name}_E_selected_kcalmol"] = e_sel
            npz_data[f"{split_name}_F_ref_mevA"] = f_ref_flat
            npz_data[f"{split_name}_F_baseline_mevA"] = f_base_flat
            npz_data[f"{split_name}_F_selected_mevA"] = f_sel_flat

            plot_energy_parity(e_ref, e_base, e_sel, f"Energy parity ({split_name} set) - {label}",
                                point_dir / f"parity_energy_{split_name}.png")
            plot_force_components(f_ref_flat, f_base_flat, f_sel_flat,
                                   f"Force component parity ({split_name} set) - {label}",
                                   point_dir / f"parity_forces_{split_name}.png")

            dev_base = directional_deviation(f_ref, f_base)
            dev_sel = directional_deviation(f_ref, f_sel)
            npz_data[f"{split_name}_directional_deviation_baseline_mevA"] = dev_base
            npz_data[f"{split_name}_directional_deviation_selected_mevA"] = dev_sel
            plot_directional(dev_base, dev_sel, f"{split_name} set - {label}",
                              point_dir / f"parity_directions_{split_name}.png")

            if has_virial:
                xi_ref = np.asarray([Xi_ref[i] for i in idx]).reshape(len(idx), -1)
                xi_base = np.asarray([Xi_base[i] for i in idx]).reshape(len(idx), -1)
                xi_sel = np.asarray([Xi_sel[i] for i in idx]).reshape(len(idx), -1)
                npz_data[f"{split_name}_Xi_ref_Ha"] = xi_ref
                npz_data[f"{split_name}_Xi_baseline_Ha"] = xi_base
                npz_data[f"{split_name}_Xi_selected_Ha"] = xi_sel
                plot_virial_parity(xi_ref.ravel(), xi_base.ravel(), xi_sel.ravel(),
                                    f"Virial parity ({split_name} set) - {label}",
                                    point_dir / f"parity_virials_{split_name}.png")

            if has_charges:
                q_ref = np.concatenate([np.asarray(Q_ref[i], dtype=float) for i in idx])
                q_base = np.concatenate([np.asarray(Q_base[i], dtype=float) for i in idx])
                q_sel = np.concatenate([np.asarray(Q_sel[i], dtype=float) for i in idx])
                npz_data[f"{split_name}_Q_ref_e"] = q_ref
                npz_data[f"{split_name}_Q_baseline_e"] = q_base
                npz_data[f"{split_name}_Q_selected_e"] = q_sel
                plot_charge_parity(q_ref, q_base, q_sel, f"Charge parity ({CHARGE_SCHEME_RUN}, {split_name} set) - {label}",
                                    point_dir / f"parity_charges_{split_name}.png")

        np.savez_compressed(point_dir / f"{label}_parity_data.npz", **npz_data)
        print(f"[{label}] Wrote parity plots + data to {point_dir}")

    compute_q_parity = False  # Q (partial charges) parity only applies to cluster Na runs with
                              # compute_charges=True; overridden below if applicable.
    Q_ref_full = Q_base_full = None

    if type_data == "pbc":
        # ----- Na-relevant PBC structures (system_filter="nacl"), saved train/test split ----- #
        pbc_dataset_dir = Path(run_config.get("dataset_path", str(PROJECT_ROOT_PATH / "structures/periodic_data/dataset")))
        pbc_poscar_cache_dir = PROJECT_ROOT_PATH / "structures/periodic_data/poscar_cache"
        pbc_system_filter = run_config.get("system_filter", "nacl")
        pbc_fit_elements = tuple(run_config.get("fit_elements", ["H", "O", "Na", "Cl"]))
        records = load_pbc_dataset(pbc_dataset_dir, system_filter=pbc_system_filter, max_structures=None)
        if PARITY_MAX_STRUCTURES is not None:
            records = records[:PARITY_MAX_STRUCTURES]
        records = ensure_poscar_cache(records, pbc_poscar_cache_dir)

        frame_id_to_pos = {r.frame_id: i for i, r in enumerate(records)}
        train_idx = np.array([frame_id_to_pos[fid] for fid in dataset_split["train_frame_ids"] if fid in frame_id_to_pos])
        test_idx = np.array([frame_id_to_pos[fid] for fid in dataset_split["test_frame_ids"] if fid in frame_id_to_pos])
        print(f"Parity dataset (PBC {pbc_system_filter}): {len(records)} structures "
              f"({len(train_idx)} train / {len(test_idx)} test, from saved dataset_split.json)")

        _idx_by_split = {"train": train_idx, "test": test_idx}
        PARITY_NEEDED_IDX = np.unique(np.concatenate(
            [_idx_by_split[name] for name in PARITY_SPLITS] or [np.array([], dtype=int)]
        ))
        print(f"Parity splits requested: {PARITY_SPLITS} -> forces/virial computed for {len(PARITY_NEEDED_IDX)}/{len(records)} "
              "structures (energy still computed for all, needed to fit train offsets)")

        def pbc_predict(param_toml_path, label, force_idx):
            """Energy for every record; forces/virial only for indices in force_idx (energy-only,
            much cheaper, elsewhere -- offsets still need every train energy regardless of splits)."""
            force_idx = set(int(i) for i in force_idx)
            E = [None] * len(records)
            F = [None] * len(records)
            Xi = [None] * len(records)

            def job(i):
                result = run_tblite_pbc(
                    poscar_path=records[i].poscar_path, nat=records[i].n_atoms,
                    param_toml_path=param_toml_path,
                    run_dir=WORKDIR / "parity_tblite" / label / records[i].frame_id,
                    charge=0, compute_forces=(i in force_idx),
                )
                return i, result.energy_ha, result.forces_ha_bohr, result.virial_ha

            with ThreadPoolExecutor(max_workers=min(PARITY_WORKERS, len(records))) as executor:
                futures = [executor.submit(job, i) for i in range(len(records))]
                for future in as_completed(futures):
                    i, e, f, xi = future.result()
                    E[i], F[i], Xi[i] = e, f, xi
            return np.asarray(E), F, Xi

        def pbc_residualize(E_full):
            """Fit composition offsets on train (per parameter set) and residualize the full set."""
            train_records = [records[i] for i in train_idx]
            ref_offsets = fit_composition_offsets(E_ref_full[train_idx], train_records, elements=pbc_fit_elements)["offsets"]
            pred_offsets = fit_composition_offsets(E_full[train_idx], train_records, elements=pbc_fit_elements)["offsets"]
            return (composition_residuals(E_ref_full, records, ref_offsets, elements=pbc_fit_elements),
                    composition_residuals(E_full, records, pred_offsets, elements=pbc_fit_elements))

        E_ref_full = np.asarray([r.ref_energy_ha for r in records])
        F_ref_full = [r.ref_forces_ha_bohr for r in records]
        Xi_ref_full = [r.ref_virial_ha for r in records]

        print("Computing baseline (default GFN2-xTB) predictions for parity plots...")
        E_base_full, F_base_full, Xi_base_full = pbc_predict(base_param_file, "parity_baseline", PARITY_NEEDED_IDX)
        E_ref_res, E_base_res = pbc_residualize(E_base_full)

    else:
        # ----- Na-relevant cluster structures (T2_Na + T3 tiers, + na_o_target/na_h2o4_target
        # targets), saved train/test split ----- #
        NA_CLUSTER_DATASET_PATH = PROJECT_ROOT_PATH / "structures/cluster_data/dataset"
        NA_CLUSTER_EXTRA_DIRS = ("na_o_target", "na_h2o4_target")
        NA_MONO_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/monomers"
        NA_MONOMER_XYZ = NA_MONO_DIR / "na/na.xyz"
        CL_MONOMER_XYZ = NA_MONO_DIR / "cl/cl.xyz"
        WATER_MONOMER_XYZ = NA_MONO_DIR / "water/water.xyz"

        def resolve_na_cluster_file(number):
            """Find the xyz for a frame number: main dataset dir first, then the na_o_target/
            na_h2o4_target subdirs (matches how nsga3_loss_regularized_Na.py assembles its
            na_coord_files)."""
            pattern_n = re.compile(rf"^{number}_.*\.xyz$")
            for subdir in (None,) + NA_CLUSTER_EXTRA_DIRS:
                d = NA_CLUSTER_DATASET_PATH if subdir is None else NA_CLUSTER_DATASET_PATH / subdir
                if not d.exists():
                    continue
                for fn in os.listdir(d):
                    if pattern_n.match(fn):
                        return d / fn, d / "dft_results_revpbe0"
            raise FileNotFoundError(f"No xyz found for frame number {number} under {NA_CLUSTER_DATASET_PATH}")

        class TBLiteMoleculeCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, param_toml_path, run_dir, charge=0, spin=None, **kwargs):
                super().__init__(**kwargs)
                self.param_toml_path = param_toml_path
                self.run_dir = Path(run_dir)
                self.charge = charge
                self.spin = spin
                self._step = 0

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                step_dir = self.run_dir / f"step_{self._step:03d}"
                self._step += 1
                step_dir.mkdir(parents=True, exist_ok=True)
                xyz_path = step_dir / "geom.xyz"
                write(xyz_path, atoms)
                energy_ha, forces_ha_bohr = run_tblite_molecule(
                    xyz_path=xyz_path, nat=len(atoms), param_toml_path=self.param_toml_path,
                    run_dir=step_dir, charge=self.charge, spin=self.spin,
                )
                self.results["energy"] = energy_ha * Hartree
                self.results["forces"] = forces_ha_bohr * (Hartree / Bohr)

        def cluster_monomer_energies_ha(param_toml_path, label):
            """(e_na, e_cl, e_water) in Ha for this parameter set, matching the RE convention of
            calculate_re_f_md_dft / the training script's compute_monomer_energies_dftbplus (e_na,
            e_cl single-point; e_water BFGS-relaxed) -- computed via tblite (this script's engine)
            rather than DFTB+ (the training engine), same simplification as test_pareto_water.py."""
            e_na, _ = run_tblite_molecule(
                xyz_path=NA_MONOMER_XYZ, nat=1, param_toml_path=param_toml_path,
                run_dir=MOLECULE_RUNS / label / "na_monomer", charge=1, compute_forces=False,
            )
            e_cl, _ = run_tblite_molecule(
                xyz_path=CL_MONOMER_XYZ, nat=1, param_toml_path=param_toml_path,
                run_dir=MOLECULE_RUNS / label / "cl_monomer", charge=-1, compute_forces=False,
            )
            water = read(WATER_MONOMER_XYZ)
            water.calc = TBLiteMoleculeCalculator(param_toml_path, MOLECULE_RUNS / label / "water_monomer_relax", charge=0)
            BFGS(water, logfile=str(RUN_OUTPUT_DIR / f"water_monomer_relax_{label}.log")).run(fmax=0.01, steps=200)
            e_water = water.get_potential_energy() / Hartree
            return e_na, e_cl, e_water

        train_numbers = list(dataset_split["train_frame_numbers"])
        test_numbers = list(dataset_split["test_frame_numbers"])
        all_numbers = train_numbers + test_numbers
        if PARITY_MAX_STRUCTURES is not None:
            keep = set(all_numbers[:PARITY_MAX_STRUCTURES])
            train_numbers = [n for n in train_numbers if n in keep]
            test_numbers = [n for n in test_numbers if n in keep]
            all_numbers = train_numbers + test_numbers

        na_cluster_files = []  # (number, xyz_path, charge, n_na, n_cl, n_waters)
        E_ref_full, F_ref_full = [], []
        for number in all_numbers:
            xyz_path, dft_dir = resolve_na_cluster_file(number)
            charge = get_total_charge(str(xyz_path))
            n_na, n_cl, n_waters = get_number_of_na_cl_waters(str(xyz_path))
            RE, F_dft, _mu = calculate_re_f_md_dft(str(xyz_path), path=str(dft_dir), verbose=False)
            na_cluster_files.append((number, xyz_path, charge, n_na, n_cl, n_waters))
            E_ref_full.append(RE)
            F_ref_full.append(np.asarray(F_dft, dtype=float))
        E_ref_full = np.asarray(E_ref_full)
        print(f"Parity dataset (cluster Na tiers): {len(na_cluster_files)} structures")

        number_to_pos = {n: i for i, (n, *_rest) in enumerate(na_cluster_files)}
        train_idx = np.array([number_to_pos[n] for n in train_numbers if n in number_to_pos])
        test_idx = np.array([number_to_pos[n] for n in test_numbers if n in number_to_pos])
        print(f"Parity split (cluster Na tiers, from saved dataset_split.json): "
              f"{len(train_idx)} train / {len(test_idx)} test")

        _idx_by_split = {"train": train_idx, "test": test_idx}
        PARITY_NEEDED_IDX = np.unique(np.concatenate(
            [_idx_by_split[name] for name in PARITY_SPLITS] or [np.array([], dtype=int)]
        ))
        print(f"Parity splits requested: {PARITY_SPLITS} -> predictions (baseline + selected) restricted to "
              f"{len(PARITY_NEEDED_IDX)}/{len(na_cluster_files)} structures")

        def cluster_predict(param_toml_path, label, only_idx=None):
            """Predict all na_cluster_files by default; only_idx restricts to a subset of positions
            (the rest stay None) -- the saved split means even the baseline pass can skip
            structures outside PARITY_SPLITS (unlike the water T1 case, no F_term-based split to
            rank first)."""
            positions = range(len(na_cluster_files)) if only_idx is None else [int(i) for i in only_idx]
            e_na, e_cl, e_water = cluster_monomer_energies_ha(param_toml_path, label)
            E = [None] * len(na_cluster_files)
            F = [None] * len(na_cluster_files)

            def job(pos):
                number, xyz_path, charge, n_na, n_cl, n_waters = na_cluster_files[pos]
                atoms = read(xyz_path)
                run_dir = MOLECULE_RUNS / label / f"cluster_{number:03d}"
                e_ha, forces_ha_bohr = run_tblite_molecule(
                    xyz_path=xyz_path, nat=len(atoms), param_toml_path=param_toml_path, run_dir=run_dir, charge=charge,
                )
                re_ha = e_ha - n_na * e_na - n_cl * e_cl - n_waters * e_water
                return pos, re_ha, forces_ha_bohr

            with ThreadPoolExecutor(max_workers=min(PARITY_WORKERS, max(len(positions), 1))) as executor:
                futures = [executor.submit(job, pos) for pos in positions]
                for future in as_completed(futures):
                    pos, e, f = future.result()
                    E[pos], F[pos] = e, f
            return E, F

        print("Computing baseline (default GFN2-xTB) predictions for parity plots...")
        E_base_full, F_base_full = cluster_predict(base_param_file, "parity_baseline", only_idx=PARITY_NEEDED_IDX)
        Xi_ref_full = Xi_base_full = None
        E_ref_res, E_base_res = E_ref_full, E_base_full  # already monomer-corrected (RE); no offsets needed

        # ----- Q (partial charges) parity -- only for runs that actually optimized Q ----- #
        compute_q_parity = bool(run_config.get("compute_charges", False)) and CHARGE_SCHEME_RUN not in (None, "N/A")
        if compute_q_parity:
            from codes.molden_handling import MoldenTemplateCache
            from codes.multiwfn_wrapper import parse_multiwfn_chg
            from codes.tblite_gradients import analyze_one_structure_tblite_multiwfn_charges

            MULTIWFN_EXE = run_config.get(
                "multiwfn_exe",
                os.environ.get("MULTIWFN_EXE", "Multiwfn"),
            )
            TEMPLATE_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/molden_templates_gfn2"
            ATMRAD_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/multiwfn_atmrad"
            MULTIWFN_THREADS_PARITY = int(os.environ.get("MULTIWFN_THREADS", 4))
            CHARGE_WORKERS = max(1, PARITY_WORKERS // MULTIWFN_THREADS_PARITY)

            print(f"Charge parity requested (scheme={CHARGE_SCHEME_RUN}); loading reference charges "
                  f"+ molden templates for {len(PARITY_NEEDED_IDX)} structures...")
            molden_template_cache = MoldenTemplateCache()
            Q_ref_full = [None] * len(na_cluster_files)
            for pos in PARITY_NEEDED_IDX:
                number, xyz_path, charge, n_na, n_cl, n_waters = na_cluster_files[pos]
                stem = xyz_path.stem

                templ_path = TEMPLATE_DIR / f"{stem}__q{charge:+d}.molden"
                if not templ_path.exists():
                    raise FileNotFoundError(f"No cached Molden template for {stem} (q={charge:+d}) at {templ_path}")
                molden_template_cache.add(stem, charge, templ_path)

                chg_path = xyz_path.parent / "dft_multiwfn_chg_revpbe0" / CHARGE_SCHEME_RUN / f"{stem}.chg"
                if not chg_path.exists():
                    raise FileNotFoundError(f"No precomputed reference charges for {stem} at {chg_path}")
                Q_ref_full[pos] = parse_multiwfn_chg(chg_path, nat=len(read(xyz_path)))

            def cluster_predict_charges(param_changes_dict, param_toml_path, label, only_idx=None):
                """Predicted partial charges (scheme=CHARGE_SCHEME_RUN) via tblite + Multiwfn,
                reusing the cached Molden templates (structure/charge only -- valid for any
                parameter set) and reconstructing the wavefunction for param_toml_path/
                param_changes_dict (the exact pairing used elsewhere for this same point)."""
                positions = range(len(na_cluster_files)) if only_idx is None else [int(i) for i in only_idx]
                Q = [None] * len(na_cluster_files)

                def job(pos):
                    _number, xyz_path, charge, _n_na, _n_cl, _n_waters = na_cluster_files[pos]
                    _re, _F, q, _molden_path = analyze_one_structure_tblite_multiwfn_charges(
                        xyz_path=xyz_path, total_charge=charge, param_toml_path=param_toml_path,
                        template_cache=molden_template_cache,
                        tblite_root=WORKDIR / "parity_multiwfn_tblite" / label,
                        molden_out_dir=WORKDIR / "parity_molden" / label,
                        multiwfn_exe=MULTIWFN_EXE, atmrad_dir=ATMRAD_DIR,
                        multiwfn_root=WORKDIR / "parity_multiwfn_runs" / label,
                        scheme=CHARGE_SCHEME_RUN, omp_multiwfn=MULTIWFN_THREADS_PARITY,
                        monomer_energies=None, param_changes=param_changes_dict,
                    )
                    return pos, q

                with ThreadPoolExecutor(max_workers=min(CHARGE_WORKERS, max(len(positions), 1))) as executor:
                    futures = [executor.submit(job, pos) for pos in positions]
                    for future in as_completed(futures):
                        pos, q = future.result()
                        Q[pos] = q
                return Q

            print("Computing baseline (default GFN2-xTB) charge predictions for parity plots...")
            Q_base_full = cluster_predict_charges({}, base_param_file, "parity_baseline_q", only_idx=PARITY_NEEDED_IDX)
        else:
            print("Charge parity skipped: this source run did not optimize Q "
                  "(compute_charges=False, or no charge scheme in run_config).")

    HAS_VIRIAL = type_data == "pbc"
    HAS_CHARGES = compute_q_parity


# ----------------------- shared helpers ----------------------- #

def fit_morse(distances_A, energies_ha):
    """Fit a Morse potential; returns (R_e [A], D_e [eV], morse_fn, popt, R_e_err [A], D_e_err [eV])."""
    distances_A = np.asarray(distances_A, dtype=float)
    shifted_ha = np.asarray(energies_ha, dtype=float) - np.min(energies_ha)

    def morse(r, d_e, a, r_e):
        return d_e * (1 - np.exp(-a * (r - r_e))) ** 2

    r_e_guess = distances_A[np.argmin(energies_ha)]
    d_e_guess = max(shifted_ha[-1], 1e-3)
    popt, pcov = curve_fit(morse, distances_A, shifted_ha, p0=[d_e_guess, 1.0, r_e_guess], maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    d_e_ha, _a, r_e = popt
    d_e_err_ha, _a_err, r_e_err = perr
    return float(r_e), float(d_e_ha) * HARTREE_TO_EV, morse, popt, float(r_e_err), float(d_e_err_ha) * HARTREE_TO_EV


def fit_re_quadratic(r_arr, e_arr):
    """R_e via local quadratic fits at ±2, ±3, ±4 around the minimum.
    Returns dict with primary R_e (±3 window), all window values, spread, and quartic diagnostic.
    """
    i_min = int(np.argmin(e_arr))
    re_windows = {}
    for hw in (2, 3, 4):
        lo, hi = max(0, i_min - hw), min(len(r_arr) - 1, i_min + hw)
        p = np.polyfit(r_arr[lo:hi + 1], e_arr[lo:hi + 1], 2)
        re_windows[f"pm{hw}"] = float(-p[1] / (2 * p[0]))
    lo4, hi4 = max(0, i_min - 5), min(len(r_arr) - 1, i_min + 5)
    p4 = np.polyfit(r_arr[lo4:hi4 + 1], e_arr[lo4:hi4 + 1], 4)
    r_dense = np.linspace(r_arr[lo4], r_arr[hi4], 1000)
    re_quartic = float(r_dense[np.argmin(np.polyval(p4, r_dense))])
    return {
        "R_e_A": re_windows["pm3"],
        "R_e_windows": re_windows,
        "window_spread_A": float(max(re_windows.values()) - min(re_windows.values())),
        "R_e_quartic_diag_A": re_quartic,
        "i_min": i_min,
    }


def label_with_stats(name, re_val, re_err, de_val, de_err):
    return f"{name} (R_e={re_val:.3f}+/-{re_err:.3f} A, D_e={de_val:.3f}+/-{de_err:.3f} eV)"


def near_only(distances, energies, max_r):
    distances = np.asarray(distances, dtype=float)
    energies = np.asarray(energies, dtype=float)
    mask = distances <= max_r
    return distances[mask], energies[mask]


def load_qchem_scan(ref_dir):
    """Load a Q-Chem scan: points.csv + calculations/*.out, return (distances_A, energies_ha)."""
    distances, energies_ha = [], []
    with open(ref_dir / "points.csv") as f:
        for row in csv.DictReader(f):
            out_path = ref_dir / "calculations" / f"{row['stem']}.out"
            matches = QCHEM_ENERGY_PATTERN.findall(out_path.read_text())
            if not matches:
                continue
            distances.append(float(row["r_angstrom"]))
            energies_ha.append(float(matches[-1]))
    return distances, energies_ha


def load_or_compute(cache_path, compute_fn, label):
    """Load (r, e_ha) from cache_path if it exists; otherwise call compute_fn(), save, return."""
    if cache_path.exists():
        d = np.load(cache_path)
        r, e = d["r"], d["e_ha"]
        print(f"  Loaded {label} curve from cache: {cache_path.name} ({len(r)} points)")
        return r, e
    print(f"  Computing {label} curve ...")
    r, e = compute_fn()
    np.savez(cache_path, r=np.asarray(r), e_ha=np.asarray(e))
    print(f"  Saved {label} curve to {cache_path.name}")
    return np.asarray(r), np.asarray(e)


# ----------------------- 2) Na+-H2O 1:1 probe setup (optional) ----------------------- #

if calculate_one_water:
    water_monomer = read(HEXAMER_DIR / "water_monomer.xyz")
    symbols = water_monomer.get_chemical_symbols()
    o_idx = symbols.index("O")
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    o_pos = water_monomer.positions[o_idx]
    away_from_h = o_pos - water_monomer.positions[h_indices].mean(axis=0)
    away_from_h /= np.linalg.norm(away_from_h)

    NEAR_R_11 = np.linspace(1.6, NEAR_MAX_R_A, 40)
    TAIL_R_11 = np.array([9.0, 10.0, 11.0, 13.0, 15.0])
    r_grid_11 = np.concatenate([NEAR_R_11, TAIL_R_11])

    geom_paths_11 = []
    for i, r in enumerate(r_grid_11):
        atoms = water_monomer.copy()
        atoms += Atoms("Na", positions=[o_pos + away_from_h * r])
        geom_dir = MOLECULE_RUNS / "na_h2o_11_geoms" / f"point_{i:02d}"
        geom_dir.mkdir(parents=True, exist_ok=True)
        xyz_path = geom_dir / "geom.xyz"
        write(xyz_path, atoms)
        geom_paths_11.append((xyz_path, len(atoms)))

    def compute_na_h2o_11_energies(param_toml_path, label):
        distances, energies_ha = [], []
        for (xyz_path, nat), r in zip(geom_paths_11, r_grid_11):
            run_dir = MOLECULE_RUNS / label / "na_h2o_11_scan" / xyz_path.parent.name
            try:
                energy_ha, _forces = run_tblite_molecule(
                    xyz_path=xyz_path, nat=nat, param_toml_path=param_toml_path,
                    run_dir=run_dir, charge=1, etemp=SCAN_ETEMP, compute_forces=False,
                )
            except subprocess.CalledProcessError:
                print(f"  [{label}] WARNING: tblite SCF failed at R={r:.3f} A; skipping point")
                continue
            distances.append(r)
            energies_ha.append(energy_ha)
        return distances, energies_ha

    # Q-Chem 1:1 reference (read once)
    r_grid_11_ref, ref_energies_11_ha = load_qchem_scan(QCHEM_11_REF_DIR)
    re_11_ref, de_11_ref, morse_fn_11_ref, popt_11_ref, re_11_ref_err, de_11_ref_err = fit_morse(r_grid_11_ref, ref_energies_11_ha)
    print(f"[Q-Chem ref 1:1] R_e = {re_11_ref:.4f} +/- {re_11_ref_err:.4f} A, D_e = {de_11_ref:.4f} +/- {de_11_ref_err:.4f} eV")

    # baseline 1:1 (cached)
    print("--- Baseline GFN2-xTB Na+-H2O 1:1 ---")
    r_grid_11_baseline, baseline_energies_11_ha = load_or_compute(
        RUN_OUTPUT_DIR / "baseline_11_curve.npz",
        lambda: compute_na_h2o_11_energies(base_param_file, "baseline_default_gfn2"),
        "baseline 1:1",
    )
    re_11_base, de_11_base, morse_fn_11_base, popt_11_base, re_11_base_err, de_11_base_err = fit_morse(
        r_grid_11_baseline, baseline_energies_11_ha
    )
    print(f"  R_e = {re_11_base:.4f} +/- {re_11_base_err:.4f} A, D_e = {de_11_base:.4f} +/- {de_11_base_err:.4f} eV")


# ----------------------- 3) Na+(H2O)4 tetrahedral rigid scan setup (optional) ----------------------- #

if calculate_four_waters:
    R_H2O4_GRID = np.linspace(1.8, 3.2, 80)

    def build_na_h2o4_scan_geoms(template_path, r_grid, geom_dir):
        """Build Na+(H2O)4 scan geometries with all Na-O distances = R.
        Each water moves rigidly along its original Na→O radial direction.
        Template must have Na first, then each water as O H H in sequence.
        """
        template = read(template_path)
        syms = template.get_chemical_symbols()
        pos = template.get_positions()
        na_idx = syms.index("Na")
        na_pos = pos[na_idx]
        waters = []
        i = 0
        while i < len(syms):
            if i == na_idx:
                i += 1
                continue
            if syms[i] == "O":
                o_pos = pos[i]
                unit_vec = (o_pos - na_pos) / np.linalg.norm(o_pos - na_pos)
                waters.append((i, i + 1, i + 2, unit_vec, pos[i + 1] - o_pos, pos[i + 2] - o_pos))
                i += 3
            else:
                i += 1
        geom_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for j, r in enumerate(r_grid):
            new_pos = pos.copy()
            for o_idx, h1_idx, h2_idx, unit_vec, h1_rel, h2_rel in waters:
                new_o = na_pos + r * unit_vec
                new_pos[o_idx] = new_o
                new_pos[h1_idx] = new_o + h1_rel
                new_pos[h2_idx] = new_o + h2_rel
            atoms = template.copy()
            atoms.set_positions(new_pos)
            xyz_path = geom_dir / f"point_{j:02d}.xyz"
            write(xyz_path, atoms)
            paths.append((xyz_path, len(atoms)))
        return paths

    def compute_na_h2o4_energies(param_toml_path, label, geom_paths):
        distances, energies_ha = [], []
        for (xyz_path, nat), r in zip(geom_paths, R_H2O4_GRID):
            run_dir = MOLECULE_RUNS / label / "na_h2o4_scan" / xyz_path.stem
            try:
                energy_ha, _forces = run_tblite_molecule(
                    xyz_path=xyz_path, nat=nat, param_toml_path=param_toml_path,
                    run_dir=run_dir, charge=1, etemp=SCAN_ETEMP, compute_forces=False,
                )
            except subprocess.CalledProcessError:
                print(f"  [{label}] WARNING: tblite SCF failed at R={r:.3f} A; skipping point")
                continue
            distances.append(r)
            energies_ha.append(energy_ha)
        return np.asarray(distances), np.asarray(energies_ha)

    na_h2o4_geoms = build_na_h2o4_scan_geoms(
        NA_H2O4_TEMPLATE, R_H2O4_GRID, MOLECULE_RUNS / "na_h2o4_scan_geoms"
    )

    # Q-Chem 4-water reference (read once)
    r_h2o4_ref_arr, e_h2o4_ref_ha_arr = [np.asarray(x) for x in load_qchem_scan(QCHEM_H2O4_REF_DIR)]
    fit_ref_4w = fit_re_quadratic(r_h2o4_ref_arr, e_h2o4_ref_ha_arr)
    print(f"[Q-Chem ref Na+(H2O)4] R_e = {fit_ref_4w['R_e_A']:.4f} A "
          f"(spread={fit_ref_4w['window_spread_A']:.4f} A, {len(r_h2o4_ref_arr)} points)")

    # baseline 4-water (cached)
    print("--- Baseline GFN2-xTB Na+(H2O)4 ---")
    r_h2o4_base, e_h2o4_base_ha = load_or_compute(
        RUN_OUTPUT_DIR / "baseline_h2o4_curve.npz",
        lambda: compute_na_h2o4_energies(base_param_file, "baseline_default_gfn2", na_h2o4_geoms),
        "baseline Na+(H2O)4",
    )
    fit_base_4w = fit_re_quadratic(r_h2o4_base, e_h2o4_base_ha)
    print(f"  R_e = {fit_base_4w['R_e_A']:.4f} A (spread={fit_base_4w['window_spread_A']:.4f} A)")


# ----------------------- per-point loop ----------------------- #

for point_idx in POINT_INDICES:
    print(f"\n{'='*60}")
    print(f"=== Pareto point {point_idx}  ({POINT_INDICES.index(point_idx)+1}/{len(POINT_INDICES)}) ===")
    print(f"{'='*60}")

    # build parameter file for this point
    if evaluate_opt_water_default_na:
        selected_vector = _default_vector
    else:
        selected_vector = X_pareto[point_idx]
    optimized_changes = vector_to_param_changes(selected_vector, structure)
    merged_changes = merge_param_changes(fixed_changes, optimized_changes)
    pt_workdir = WORKDIR / f"point_{point_idx:03d}"
    param_toml_path_pt = ensure_param_file(merged_changes, param_file=str(base_param_file), DIR=pt_workdir)
    print(f"Selected Na parameters: {dict(zip(parameter_names, selected_vector))}")

    label_pt = f"point_{point_idx:03d}"
    print_bound_proximity(parameter_names, selected_vector, _bounds, label_pt)

    if PLOT_PARITY:
        print(f"[{label_pt}] Computing parity-plot predictions...")
        if type_data == "pbc":
            E_sel_full, F_sel_full, Xi_sel_full = pbc_predict(param_toml_path_pt, label_pt, PARITY_NEEDED_IDX)
            _, E_sel_res = pbc_residualize(E_sel_full)
            Q_sel_full = None
        else:
            E_sel_res, F_sel_full = cluster_predict(param_toml_path_pt, label_pt, only_idx=PARITY_NEEDED_IDX)
            Xi_sel_full = None
            Q_sel_full = None
            if HAS_CHARGES:
                Q_sel_full = cluster_predict_charges(merged_changes, param_toml_path_pt, label_pt, only_idx=PARITY_NEEDED_IDX)
        parity_plots_for_point(
            label_pt, PARITY_DIR / label_pt,
            E_ref_res, F_ref_full, Xi_ref_full, Q_ref_full,
            E_base_res, F_base_full, Xi_base_full, Q_base_full,
            E_sel_res, F_sel_full, Xi_sel_full, Q_sel_full,
            train_idx, test_idx, HAS_VIRIAL, HAS_CHARGES, PARITY_SPLITS,
        )

    # --- 1:1 selected (cached, optional) ---
    if calculate_one_water:
        r_grid_11_sel, e_11_sel_ha = load_or_compute(
            RUN_OUTPUT_DIR / f"{label_pt}_11_curve.npz",
            lambda: compute_na_h2o_11_energies(param_toml_path_pt, label_pt),
            f"{label_pt} 1:1",
        )
        re_11, de_11, morse_fn_11, popt_11, re_11_err, de_11_err = fit_morse(r_grid_11_sel, e_11_sel_ha)
        print(f"[{label_pt}] Na+-H2O (1:1): R_e = {re_11:.4f} +/- {re_11_err:.4f} A, D_e = {de_11:.4f} +/- {de_11_err:.4f} eV")

    # --- 4-water selected (cached) ---
    if calculate_four_waters:
        r_h2o4_sel, e_h2o4_sel_ha = load_or_compute(
            RUN_OUTPUT_DIR / f"{label_pt}_h2o4_curve.npz",
            lambda: compute_na_h2o4_energies(param_toml_path_pt, label_pt, na_h2o4_geoms),
            f"{label_pt} Na+(H2O)4",
        )
        fit_sel_4w = fit_re_quadratic(r_h2o4_sel, e_h2o4_sel_ha)
        delta_re_4w = fit_sel_4w["R_e_A"] - fit_base_4w["R_e_A"]
        print(f"[{label_pt}] Na+(H2O)4: R_e = {fit_sel_4w['R_e_A']:.4f} A "
              f"(spread={fit_sel_4w['window_spread_A']:.4f} A)")
        print(f"[{label_pt}] Na+(H2O)4: Delta R_e = {delta_re_4w:+.4f} A "
              f"{'(outward shift -- good)' if delta_re_4w > 0 else '(inward shift)'}")

    # --- save results ---
    results = {
        "source_run_id": SOURCE_RUN_ID,
        "point_index": point_idx,
        "charge_scheme": CHARGE_SCHEME,
        "parameter_names": parameter_names,
        "parameters": {n: float(v) for n, v in zip(parameter_names, selected_vector)},
        "selected": {},
        "baseline_default_gfn2": {},
    }
    if calculate_one_water:
        results["selected"]["na_h2o_1to1"] = {
            "R_e_A": re_11, "R_e_err_A": re_11_err,
            "D_e_eV": de_11, "D_e_err_eV": de_11_err,
            "n_converged": len(r_grid_11_sel), "n_fit_grid": len(r_grid_11),
        }
        results["baseline_default_gfn2"]["na_h2o_1to1"] = {
            "R_e_A": re_11_base, "R_e_err_A": re_11_base_err,
            "D_e_eV": de_11_base, "D_e_err_eV": de_11_base_err,
            "n_converged": len(r_grid_11_baseline), "n_fit_grid": len(r_grid_11),
        }
        results["na_h2o_1to1_qchem_reference"] = {
            "R_e_A": re_11_ref, "R_e_err_A": re_11_ref_err,
            "D_e_eV": de_11_ref, "D_e_err_eV": de_11_ref_err,
            "n_points": len(r_grid_11_ref),
        }
    if calculate_four_waters:
        results["na_h2o4_tetrahedral_rigid_scan"] = {
            "selected": {k: v for k, v in fit_sel_4w.items() if k != "i_min"},
            "baseline_default_gfn2": {k: v for k, v in fit_base_4w.items() if k != "i_min"},
            "qchem_reference": {k: v for k, v in fit_ref_4w.items() if k != "i_min"},
            "delta_R_e_A": delta_re_4w,
        }
    (RUN_OUTPUT_DIR / f"results_idx_{point_idx:03d}.json").write_text(json.dumps(results, indent=2))

    # --- plot 1: 1:1 probe (optional) ---
    plot1_panels = []
    if calculate_one_water:
        plot1_panels.append("11")

    if plot1_panels:
        fig, axes = plt.subplots(1, len(plot1_panels), figsize=(5 * len(plot1_panels), 4.5))
        axes = np.atleast_1d(axes)
        ax_by_panel = dict(zip(plot1_panels, axes))

        if calculate_one_water:
            ax = ax_by_panel["11"]
            gfn2_11_min = np.min(e_11_sel_ha)
            baseline_11_min = np.min(baseline_energies_11_ha)
            ref_11_min = np.min(ref_energies_11_ha)
            r_sel_near, e_sel_near = near_only(r_grid_11_sel, e_11_sel_ha, NEAR_MAX_R_A)
            r_base_near, e_base_near = near_only(r_grid_11_baseline, baseline_energies_11_ha, NEAR_MAX_R_A)
            r_ref_near, e_ref_near = near_only(r_grid_11_ref, ref_energies_11_ha, NEAR_MAX_R_A)
            ax.plot(r_sel_near, e_sel_near - gfn2_11_min, "o", ms=3, color="C0")
            ax.plot(r_base_near, e_base_near - baseline_11_min, "s", ms=3, color="0.5")
            ax.plot(r_ref_near, e_ref_near - ref_11_min, "x", ms=4, color="C2")
            r_fit_11 = np.linspace(NEAR_R_11.min(), NEAR_MAX_R_A, 300)
            ax.plot(r_fit_11, morse_fn_11(r_fit_11, *popt_11), "-", color="C0",
                    label=label_with_stats(OUR_LABEL, re_11, re_11_err, de_11, de_11_err))
            ax.plot(r_fit_11, morse_fn_11_base(r_fit_11, *popt_11_base), "--", color="0.5",
                    label=label_with_stats("GFN2-xTB (default)", re_11_base, re_11_base_err, de_11_base, de_11_base_err))
            ax.plot(r_fit_11, morse_fn_11_ref(r_fit_11, *popt_11_ref), ":", color="C2",
                    label=label_with_stats("Q-Chem revPBE0/aug-cc-pVTZ", re_11_ref, re_11_ref_err, de_11_ref, de_11_ref_err))
            ax.set_xlim(right=NEAR_MAX_R_A)
            ax.set_xlabel("Na-O distance (A)")
            ax.set_ylabel("Energy (Ha, shifted)")
            ax.set_title("Na+-H2O (1:1) probe")
            ax.legend(fontsize=6)

        fig.suptitle(f"idx {point_idx}")
        fig.tight_layout()
        fig.savefig(RUN_OUTPUT_DIR / f"na_water_overbinding_idx_{point_idx:03d}.png", dpi=150)
        plt.close(fig)

    # --- plot 2: 4-water scan ---
    if calculate_four_waters:
        e_sel_shifted = e_h2o4_sel_ha - e_h2o4_sel_ha.min()
        e_base_shifted = e_h2o4_base_ha - e_h2o4_base_ha.min()
        e_ref_shifted = e_h2o4_ref_ha_arr - e_h2o4_ref_ha_arr.min()

        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        ax4.plot(r_h2o4_sel, e_sel_shifted, "o", ms=3, color="C0")
        ax4.plot(r_h2o4_base, e_base_shifted, "s", ms=3, color="0.5")
        ax4.plot(r_h2o4_ref_arr, e_ref_shifted, "x", ms=4, color="C2")

        for r_arr, e_shifted, fit, color, ls, name in [
            (r_h2o4_sel,      e_sel_shifted,  fit_sel_4w,  "C0",  "-",  OUR_LABEL),
            (r_h2o4_base,     e_base_shifted, fit_base_4w, "0.5", "--", "GFN2-xTB (default)"),
            (r_h2o4_ref_arr,  e_ref_shifted,  fit_ref_4w,  "C2",  ":",  "Q-Chem revPBE0/aug-cc-pVTZ"),
        ]:
            i_min = fit["i_min"]
            lo, hi = max(0, i_min - 3), min(len(r_arr) - 1, i_min + 3)
            p = np.polyfit(r_arr[lo:hi + 1], e_shifted[lo:hi + 1], 2)
            r_fit_4w = np.linspace(r_arr[lo], r_arr[hi], 200)
            re = fit["R_e_A"]
            spread = fit["window_spread_A"]
            ax4.plot(r_fit_4w, np.polyval(p, r_fit_4w), ls, color=color,
                     label=f"{name}: R_e={re:.3f} A (spread={spread:.4f} A)")
            ax4.axvline(re, color=color, lw=0.8, ls=":")

        ax4.set_xlabel("Na-O distance (A)")
        ax4.set_ylabel("Energy (Ha, shifted)")
        ax4.set_title(f"Na+(H2O)4 tetrahedral rigid scan  |  idx {point_idx}  |  "
                      f"ΔR_e(sel-def) = {delta_re_4w:+.3f} A")
        ax4.legend(fontsize=8)
        fig4.tight_layout()
        fig4.savefig(RUN_OUTPUT_DIR / f"na_h2o4_tetrahedral_scan_idx_{point_idx:03d}.png", dpi=150)
        plt.close(fig4)

print(f"\nWrote results to {RUN_OUTPUT_DIR}")
