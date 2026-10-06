"""
Validate a selected H/O Pareto point against basic molecular properties:

  - equilibrium O-H bond length and (first) O-H bond dissociation energy of
    an isolated water monomer, via H2O -> OH(doublet) + H(atom), compared
    against a Q-Chem revPBE0/aug-cc-pVTZ reference
    (q-chem-codes/xtb-project/sm_test/water_monomer_bde_revpbe0_augccpvtz)
  - water hexamer (book/cage/prism/ring) binding energies and ordering vs
    CCSD(T)/AVDZ references (Table III of https://doi.org/10.1063/1.4922262)
    and a Q-Chem revPBE0/aug-cc-pVTZ reference at the same fixed geometries
    (q-chem-codes/xtb-project/sm_test/water_hexamers_revpbe0_augccpvtz)

Which Pareto run/points to test is hardcoded below (SOURCE_RUN_ID,
POINT_INDICES) -- no CLI arguments are used for science settings, only for HPC
run plumbing, matching the optimizer scripts' convention.

POINT_INDICES is a list; one set of output files (results_idx_*.json,
hexamer_binding_idx_*.png) is written per index. The baseline GFN2-xTB
properties are computed once and reused for every point.

Submit with:

  sbatch optimization/analyze_pareto/submit_test_pareto_water.slurm
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.environ.get('USER', 'user')}"))

import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import Bohr, Hartree
from sklearn.metrics import r2_score, root_mean_squared_error

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
PROJECT_ROOT = str(PROJECT_ROOT_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from codes.calculate_ae_charged import vector_to_param_changes
from codes.gradients import ensure_param_file, get_base_param_file
from codes.gradients_qchem import calculate_re_f_md_dft
from codes.pbc_data import composition_residuals, ensure_poscar_cache, fit_composition_offsets, load_pbc_dataset
from codes.pbc_tblite import get_or_run_isolated_atom_energies, run_tblite_molecule, run_tblite_pbc
from codes.common import (
    init_tblite_env,
    generate_parameter_info,
)


# ----------------------- what to test ----------------------- #

#SOURCE_RUN_ID = "nsga3_pbc_efxi_loss_210083_20260529_092923"
#POINT_INDEX = 11

# cluster data
'''
SOURCE_RUN_ID = "nsga_t1_ef_water_215904_20260720_101414"
POINT_INDICES = [  0,   2,   4,  15,  18,  19,  20,  21,  27,  29,  31,  32,  34,
        35,  39,  41,  43,  49,  51,  56,  61,  64,  66,  68,  69,  71,
        78,  82,  83,  84,  85,  88,  90,  92,  94,  96,  97, 102, 103,
       108, 111, 113, 116, 117, 118, 119, 121, 130, 135, 137, 141, 145,
       146, 147, 148, 149, 150, 152, 157, 160, 161, 163, 165, 171, 172,
       173, 176, 178, 182, 186, 188, 189, 190, 192, 197, 211, 215, 217,
       223, 227, 230, 233, 235, 236, 240, 242, 243, 244, 247, 249, 252,
       254, 259, 261, 262, 263, 264, 266, 267, 269, 272, 274, 276, 277,
       279, 282, 283, 286, 287, 288, 294, 296, 301, 303, 307, 309, 311,
       316, 320, 323, 324, 326, 329, 331, 333, 336, 337, 339, 342, 343,
       345, 350, 356, 357, 358, 359, 360, 361, 363, 364, 365, 366, 370,
       372, 373, 375, 379, 381, 382, 385, 387, 389, 392, 397, 398, 402,
       406, 415, 417, 418, 420, 423, 427, 429, 430, 432, 433, 434, 436,
       437, 438]
'''
#SOURCE_RUN_ID = "nsga_t1_ef_water_209526_20260522_121543"
#POINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

#SOURCE_RUN_ID = "nsga_t1_ef_water_215904_20260720_101414"
#POINT_INDICES = [249, 252]

#SOURCE_RUN_ID = 'nsga3_pbc_loss_208819_20260516_180923'
#POINT_INDICES = [4, 5]

#SOURCE_RUN_ID = 'nsga3_pbc_efxi_loss_210083_20260529_092923'
#POINT_INDICES = [3, 37, 11]

SOURCE_RUN_ID = 'nsga3_pbc_efxi_loss_215185_20260702_084458' #PBC (E,F)
POINT_INDICES = [30]

# If True, additionally evaluate E/F/(Xi, PBC only) parity against the same train/test split
# used by the original optimization script (nsga3_EF_water.py for cluster, nsga3_pbc_loss_regularized.py
# for PBC), for default GFN2-xTB and every selected point, under RUN_OUTPUT_DIR/parity_plots/.
# If False, the script behaves exactly as before. This is expensive (a full tblite sweep over
# the train+test split per point) -- PARITY_MAX_STRUCTURES caps it for a quick look.
PLOT_PARITY = True
PARITY_MAX_STRUCTURES = None  # debug knob: cap #structures (train+test) used for parity; None = full split
PARITY_SPLITS = ("test",)  # which split(s) to actually plot/save: subset of ("train", "test").
                            # Structures outside this set are never predicted for cluster data, and are
                            # predicted energy-only (no forces/virial) for PBC data (offsets still need
                            # train energies) -- both cut compute substantially vs. ("train", "test").


HEXAMER_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/water_hexamers"
REF_BINDING_ENERGIES_KCAL = {  # CCSD(T)/AVDZ, Table III of https://doi.org/10.1063/1.4922262
    "book": -47.671,
    "cage": -48.352,
    "prism": -48.793,
    "ring": -46.123,
}

QCHEM_MONOMER_BDE_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/water_monomer_bde"
QCHEM_HEXAMERS_DIR = PROJECT_ROOT_PATH / "structures/cluster_data/water_hexamers"
DFT_MONOMER = json.loads((QCHEM_MONOMER_BDE_DIR / "results.json").read_text())
DFT_HEXAMERS = json.loads((QCHEM_HEXAMERS_DIR / "results.json").read_text())
print(f"Loaded DFT (revPBE0/aug-cc-pVTZ) monomer reference: {QCHEM_MONOMER_BDE_DIR / 'results.json'}")
print(f"Loaded DFT (revPBE0/aug-cc-pVTZ) hexamer reference: {QCHEM_HEXAMERS_DIR / 'results.json'}")

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.503


# ----------------------- HPC run plumbing (same pattern as the optimizers) ----------------------- #


init_tblite_env(omp_threads=1, blas_threads=1, amd_mkl=True)


def _default_scratch_root():
    candidates = [Path(p) for p in (os.environ.get("SCRATCH"), "/tmp") if p]
    for path in candidates:
        if path.exists() and os.access(path, os.W_OK):
            return path
    return Path("/tmp")


DEFAULT_SCRATCH_ROOT = _default_scratch_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT_PATH / "optimization/analyze_pareto/pareto_tests"


def _default_run_id():
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if slurm_job_id:
        return f"test_pareto_water_{slurm_job_id}_{stamp}"
    return f"test_pareto_water_local_{stamp}"


def parse_hpc_args():
    parser = argparse.ArgumentParser(description="Validate a water Pareto point's molecular properties.")
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
ATOM_RUNS = WORKDIR / "isolated_atoms_tblite"
MOLECULE_RUNS = WORKDIR / "molecule_tblite_runs"

WORKDIR.mkdir(parents=True, exist_ok=True)
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"RUN_ID: {RUN_ID}")
print(f"Scratch workdir: {WORKDIR}")
print(f"Persistent output dir: {RUN_OUTPUT_DIR}")
print(f"Source Pareto run: {SOURCE_RUN_ID}, point indices: {POINT_INDICES}")


# ----------------------- load the selected Pareto point ----------------------- #
type_data = "pbc" if SOURCE_RUN_ID.startswith("nsga3_pbc_") else "cluster"

# Two known H/O parameter schemas across the run history -- match specific legacy run IDs
# *before* the generic prefix checks below, since a legacy run's ID can still start with
# "nsga3_pbc_" / match the generic cluster pattern and would otherwise be shadowed.
LEGACY_PARAM_CHANGES = {  # pre "H/O parameter fix" runs (12 params: gam/lgam/gam3 included)
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

if SOURCE_RUN_ID == 'nsga3_pbc_loss_208819_20260516_180923':
    SOURCE_RUN_DIR = PROJECT_ROOT_PATH / "optimization/T_pbc/hpc_runs" / SOURCE_RUN_ID
    run_config = json.loads((SOURCE_RUN_DIR / "run_config.json").read_text())
    npz_path = next(SOURCE_RUN_DIR.glob(f"*results_pbc*{SOURCE_RUN_ID}.npz"))
    param_changes = LEGACY_PARAM_CHANGES
elif SOURCE_RUN_ID == "nsga_t1_ef_water_209526_20260522_121543":
    SOURCE_RUN_DIR = PROJECT_ROOT_PATH / "optimization/T1_partial_charge_schemes/hpc_runs" / SOURCE_RUN_ID
    run_config = json.loads((SOURCE_RUN_DIR / "run_config.json").read_text())
    npz_path = next(SOURCE_RUN_DIR.glob(f"*results*{SOURCE_RUN_ID}.npz"))
    param_changes = LEGACY_PARAM_CHANGES
elif SOURCE_RUN_ID.startswith('nsga3_pbc_'):
    SOURCE_RUN_DIR = PROJECT_ROOT_PATH / "optimization/T_pbc/hpc_runs" / SOURCE_RUN_ID
    run_config = json.loads((SOURCE_RUN_DIR / "run_config.json").read_text())
    npz_path = next(SOURCE_RUN_DIR.glob(f"*results_pbc*{SOURCE_RUN_ID}.npz"))
    # Same param_changes / generate_parameter_info as nsga3_pbc_loss_regularized.py
    param_changes = {
        "kpair": {
            "O-O": 1.0,
            "O-H": 1.0,
        },
        "H": {
            "zeff": 1.1053880000000000,
            "arep": 2.2137169999999999,
        },
        "O": {
            "zeff": 5.7844150000000001,
            "arep": 2.1657120000000001,
        },
    }
else:
    SOURCE_RUN_DIR = PROJECT_ROOT_PATH / "optimization/T1_partial_charge_schemes/hpc_runs" / SOURCE_RUN_ID
    run_config = json.loads((SOURCE_RUN_DIR / "run_config.json").read_text())
    npz_path = next(SOURCE_RUN_DIR.glob(f"*results*{SOURCE_RUN_ID}.npz"))
    param_changes = {
        "kpair": {
            "O-O": 1.0,
            "O-H": 1.0,
        },
        "H": {
            "zeff": 1.1053880000000000,
            "arep": 2.2137169999999999,
        },
        "O": {
            "zeff": 5.7844150000000001,
            "arep": 2.1657120000000001,
        },
    }
pareto = np.load(npz_path, allow_pickle=True)
X_pareto = pareto["X_pareto"]
print(f"Loaded Pareto front: {npz_path.name}")

base_param_file = Path(run_config.get("base_param_file", str(get_base_param_file())))
if not base_param_file.exists():
    base_param_file = get_base_param_file()
print(f"Base param file: {base_param_file}")




names, _bounds, _default_vector, structure = generate_parameter_info(param_changes, fixed_components=None)
if "parameter_names" in pareto.files:
    parameter_names = [str(n) for n in pareto["parameter_names"]]
    assert names == parameter_names, f"Parameter order mismatch: {names} vs {parameter_names}"
else:
    print(f"Note: {npz_path.name} has no 'parameter_names' array (older run format); "
          "using names derived from param_changes instead.")
    parameter_names = names
print(f"Parameter names: {parameter_names}")


# ----------------------- molecular property evaluation ----------------------- #

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
            xyz_path=xyz_path,
            nat=len(atoms),
            param_toml_path=self.param_toml_path,
            run_dir=step_dir,
            charge=self.charge,
            spin=self.spin,
        )
        self.results["energy"] = energy_ha * Hartree
        self.results["forces"] = forces_ha_bohr * (Hartree / Bohr)


monomer_path = HEXAMER_DIR / "water_monomer.xyz"
isomers = ["book", "cage", "prism", "ring"]
ref_order = sorted(isomers, key=lambda n: REF_BINDING_ENERGIES_KCAL[n])


def molecule_energy_ha(xyz_path, param_toml_path, run_root, charge=0):
    atoms = read(xyz_path)
    run_dir = run_root / Path(xyz_path).stem
    energy_ha, _forces = run_tblite_molecule(
        xyz_path=xyz_path,
        nat=len(atoms),
        param_toml_path=param_toml_path,
        run_dir=run_dir,
        charge=charge,
    )
    return energy_ha


def evaluate_water_properties(param_toml_path, label):
    """O-H equilibrium bond length, (first) O-H bond dissociation energy, and
    hexamer binding energies/ordering for a given GFN2-xTB parameter file.

    Dissociation energy is computed along H2O -> OH(doublet) + H(atom):

      DE = E_OH,opt + E_H,opt - E_H2O,opt
    """
    run_root = MOLECULE_RUNS / label

    water = read(monomer_path)
    water.calc = TBLiteMoleculeCalculator(param_toml_path, run_root / "water_relax", charge=0)
    BFGS(water, logfile=str(RUN_OUTPUT_DIR / f"water_relax_{label}.log")).run(fmax=0.01, steps=200)

    oh_distances = [float(water.get_distance(0, 1)), float(water.get_distance(0, 2))]
    oh_mean = float(np.mean(oh_distances))
    e_water_relaxed_ha = water.get_potential_energy() / Hartree

    oh_radical = read(monomer_path)
    del oh_radical[2]  # drop one H -> OH radical (doublet, 1 unpaired electron)
    oh_radical.calc = TBLiteMoleculeCalculator(param_toml_path, run_root / "oh_radical_relax", charge=0, spin=1)
    BFGS(oh_radical, logfile=str(RUN_OUTPUT_DIR / f"oh_radical_relax_{label}.log")).run(fmax=0.01, steps=200)
    e_oh_radical_ha = oh_radical.get_potential_energy() / Hartree

    atom_energies = get_or_run_isolated_atom_energies(
        symbols=["H"],
        param_toml_path=param_toml_path,
        cache_root=ATOM_RUNS,
    )
    e_h = atom_energies["H"]["energy_ha"]
    dissociation_energy_ev = (e_oh_radical_ha + e_h - e_water_relaxed_ha) * HARTREE_TO_EV

    e_monomer_sp_ha = molecule_energy_ha(monomer_path, param_toml_path, run_root)
    binding_kcal = {}
    for name in isomers:
        xyz_path = HEXAMER_DIR / f"H2O_6_{name}.xyz"
        e_hex_ha = molecule_energy_ha(xyz_path, param_toml_path, run_root)
        binding_kcal[name] = (e_hex_ha - 6 * e_monomer_sp_ha) * HARTREE_TO_KCAL
    gfn2_order = sorted(isomers, key=lambda n: binding_kcal[n])

    dft_oh_mean = DFT_MONOMER["oh_distance_mean_A"]
    dft_diss_ev = DFT_MONOMER["dissociation_energy_eV"]
    delta_oh_vs_dft = oh_mean - dft_oh_mean
    delta_diss_vs_dft = dissociation_energy_ev - dft_diss_ev

    print(f"[{label}] relaxed O-H distances: {oh_distances} A (mean {oh_mean:.4f} A)  "
          f"[DFT revPBE0/aug-cc-pVTZ = {dft_oh_mean:.4f} A, delta = {delta_oh_vs_dft:+.4f} A]")
    print(f"[{label}] O-H bond dissociation energy (H2O -> OH + H): {dissociation_energy_ev:.4f} eV  "
          f"[DFT revPBE0/aug-cc-pVTZ = {dft_diss_ev:.4f} eV, delta = {delta_diss_vs_dft:+.4f} eV]")
    print(f"[{label}] hexamer binding energies (kcal/mol):")
    for name in isomers:
        dft_kcal = DFT_HEXAMERS["hexamer_binding_kcal_mol"][name]
        print(f"  [{label}] {name:6s} = {binding_kcal[name]:8.3f}   "
              f"DFT = {dft_kcal:8.3f}   CCSD(T)/AVDZ = {REF_BINDING_ENERGIES_KCAL[name]:8.3f}")
    print(f"[{label}] ordering (most to least bound): {gfn2_order}")
    print(f"[{label}] ordering matches CCSD(T)/AVDZ reference: {gfn2_order == ref_order}")
    print(f"[{label}] ordering matches DFT reference: {gfn2_order == DFT_HEXAMERS['qchem_ordering']}")

    return {
        "oh_distances_A": oh_distances,
        "oh_distance_mean_A": oh_mean,
        "oh_distance_mean_vs_dft_A": delta_oh_vs_dft,
        "dissociation_energy_eV": dissociation_energy_ev,
        "dissociation_energy_vs_dft_eV": delta_diss_vs_dft,
        "hexamer_binding_kcal_mol": binding_kcal,
        "gfn2_ordering": gfn2_order,
        "ordering_matches_reference": gfn2_order == ref_order,
        "ordering_matches_dft": gfn2_order == DFT_HEXAMERS["qchem_ordering"],
        "e_water_relaxed_ha": e_water_relaxed_ha,
    }


baseline = evaluate_water_properties(base_param_file, "baseline_default_gfn2")


# ----------------------- parity plots (optional, PLOT_PARITY) ----------------------- #
#
# Reproduces the E/F/(Xi) parity-plot style of the original optimization scripts
# (nsga3_EF_water.py for cluster data, nsga3_pbc_loss_regularized.py for PBC data), on the
# *same* train/test split those scripts use, comparing default GFN2-xTB vs each selected
# Pareto point. Virial (Xi) parity only applies to PBC data -- clusters have no defined
# virial/stress tensor (see fim_EF_water.py).

if PLOT_PARITY:
    PARITY_DIR = RUN_OUTPUT_DIR / "parity_plots"
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    PARITY_WORKERS = int(os.environ.get("PARITY_WORKERS", 32))
    FORCE_HA_BOHR_TO_MEV_A = 1000.0 * HARTREE_TO_EV / 0.529177210903

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

    def parity_plots_for_point(label, point_dir, E_ref, F_ref, Xi_ref, E_base, F_base, Xi_base,
                                E_sel, F_sel, Xi_sel, train_idx, test_idx, has_virial, splits):
        """Plot + save (single npz) E/F/(Xi) parity for one Pareto point, on the requested splits.

        E_*/F_*/Xi_* may be plain lists with None entries outside `splits` (structures that
        were skipped, or predicted energy-only) -- always indexed element-wise, never bulk
        np.asarray()'d, so those None entries are never touched as long as `splits` matches
        what was actually predicted upstream.
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

        np.savez_compressed(point_dir / f"{label}_parity_data.npz", **npz_data)
        print(f"[{label}] Wrote parity plots + data to {point_dir}")

    if type_data == "pbc":
        # ----- same random 70/30 water split as nsga3_pbc_loss_regularized.py ----- #
        pbc_dataset_dir = PROJECT_ROOT_PATH / "structures/periodic_data/dataset"
        pbc_poscar_cache_dir = PROJECT_ROOT_PATH / "structures/periodic_data/poscar_cache"
        records = load_pbc_dataset(pbc_dataset_dir, system_filter="water", max_structures=None)
        if PARITY_MAX_STRUCTURES is not None:
            records = records[:PARITY_MAX_STRUCTURES]
        records = ensure_poscar_cache(records, pbc_poscar_cache_dir)

        split_seed, train_frac = 123, 0.70  # matches nsga3_pbc_loss_regularized.py
        rng = np.random.default_rng(split_seed)
        shuffled = rng.permutation(len(records))
        train_set = set(int(i) for i in shuffled[: int(round(train_frac * len(records)))])
        train_idx = np.array([i for i in range(len(records)) if i in train_set])
        test_idx = np.array([i for i in range(len(records)) if i not in train_set])
        print(f"Parity dataset (PBC water): {len(records)} structures ({len(train_idx)} train / {len(test_idx)} test)")

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
            ref_offsets = fit_composition_offsets(E_ref_full[train_idx], train_records)["offsets"]
            pred_offsets = fit_composition_offsets(E_full[train_idx], train_records)["offsets"]
            return composition_residuals(E_ref_full, records, ref_offsets), composition_residuals(E_full, records, pred_offsets)

        E_ref_full = np.asarray([r.ref_energy_ha for r in records])
        F_ref_full = [r.ref_forces_ha_bohr for r in records]
        Xi_ref_full = [r.ref_virial_ha for r in records]

        print("Computing baseline (default GFN2-xTB) predictions for parity plots...")
        E_base_full, F_base_full, Xi_base_full = pbc_predict(base_param_file, "parity_baseline", PARITY_NEEDED_IDX)
        E_ref_res, E_base_res = pbc_residualize(E_base_full)

    else:
        # ----- same F_term-stratified T1 split as nsga3_EF_water.py's split_train_test() ----- #
        cluster_dataset_path = PROJECT_ROOT_PATH / "structures/cluster_data/dataset"
        cluster_dft_path = cluster_dataset_path / "dft_results_revpbe0"
        pattern = re.compile(r"^(\d+)_.*\.xyz$")
        cluster_files = []
        for fn in os.listdir(cluster_dataset_path):
            m = pattern.match(fn)
            if m and 1 <= int(m.group(1)) <= 81:  # tier T1 only (neutral water clusters)
                cluster_files.append((int(m.group(1)), fn))
        cluster_files.sort(key=lambda item: item[0])
        if PARITY_MAX_STRUCTURES is not None:
            cluster_files = cluster_files[:PARITY_MAX_STRUCTURES]

        E_ref_full, F_ref_full, signs = [], [], []
        for _file_number, filename in cluster_files:
            RE, F_dft, _mu = calculate_re_f_md_dft(str(cluster_dataset_path / filename), path=str(cluster_dft_path), verbose=False)
            E_ref_full.append(RE)
            F_ref_full.append(np.asarray(F_dft, dtype=float))
            base = filename.split("_", 1)[1].rsplit(".", 1)[0]
            signs.append("eq" if "mode" not in base else ("minus" if "minus" in base else "plus"))
        E_ref_full = np.asarray(E_ref_full)
        print(f"Parity dataset (cluster T1 water): {len(cluster_files)} structures")

        def cluster_predict(param_toml_path, e_water_relaxed_ha, label, only_idx=None):
            """Predict all cluster_files by default; only_idx restricts to a subset of positions
            (the rest stay None) -- used to skip structures outside PARITY_SPLITS entirely."""
            positions = range(len(cluster_files)) if only_idx is None else [int(i) for i in only_idx]
            E = [None] * len(cluster_files)
            F = [None] * len(cluster_files)

            def job(pos):
                file_number, filename = cluster_files[pos]
                xyz_path = cluster_dataset_path / filename
                atoms = read(xyz_path)
                n_waters = atoms.get_chemical_symbols().count("O")
                run_dir = MOLECULE_RUNS / label / f"cluster_{file_number:03d}"
                e_ha, forces_ha_bohr = run_tblite_molecule(
                    xyz_path=xyz_path, nat=len(atoms), param_toml_path=param_toml_path, run_dir=run_dir, charge=0,
                )
                return pos, e_ha - n_waters * e_water_relaxed_ha, forces_ha_bohr

            with ThreadPoolExecutor(max_workers=min(PARITY_WORKERS, max(len(positions), 1))) as executor:
                futures = [executor.submit(job, pos) for pos in positions]
                for future in as_completed(futures):
                    pos, e, f = future.result()
                    E[pos], F[pos] = e, f
            return E, F

        print("Computing baseline (default GFN2-xTB) predictions for parity plots...")
        E_base_full, F_base_full = cluster_predict(base_param_file, baseline["e_water_relaxed_ha"], "parity_baseline")
        Xi_ref_full = Xi_base_full = None
        E_ref_res, E_base_res = E_ref_full, E_base_full  # already monomer-corrected (RE); no offsets needed

        # T1 is in eq_to_train_tiers, so 'eq' structures always stay in train; the rest are
        # ranked by per-structure baseline force MSE (F_term) and spread evenly into test_frac.
        is_eq = np.asarray(signs) == "eq"
        candidate_idx = np.flatnonzero(~is_eq)
        F_term = np.asarray([float(np.sum((fp - fr) ** 2)) / (3 * len(fr)) for fr, fp in zip(F_ref_full, F_base_full)])
        n_test = max(1, min(round(0.25 * len(cluster_files)), len(candidate_idx)))
        order = candidate_idx[np.argsort(F_term[candidate_idx])]
        test_idx = np.sort(order[np.unique(np.linspace(0, len(order) - 1, num=n_test, dtype=int))])
        train_idx = np.setdiff1d(np.arange(len(cluster_files)), test_idx)
        print(f"Parity split (T1 water, F_term-stratified): {len(train_idx)} train / {len(test_idx)} test")

        _idx_by_split = {"train": train_idx, "test": test_idx}
        PARITY_NEEDED_IDX = np.unique(np.concatenate(
            [_idx_by_split[name] for name in PARITY_SPLITS] or [np.array([], dtype=int)]
        ))
        print(f"Parity splits requested: {PARITY_SPLITS} -> selected-point predictions restricted to "
              f"{len(PARITY_NEEDED_IDX)}/{len(cluster_files)} structures "
              "(baseline itself always needs the full T1 set, to rank the F_term-stratified split)")

    HAS_VIRIAL = type_data == "pbc"


# ----------------------- per-point loop ----------------------- #

for point_idx in POINT_INDICES:
    print(f"\n{'='*60}")
    print(f"=== Pareto point {point_idx}  ({POINT_INDICES.index(point_idx)+1}/{len(POINT_INDICES)}) ===")
    print(f"{'='*60}")

    selected_vector = X_pareto[point_idx]
    param_dict = vector_to_param_changes(selected_vector, structure)
    pt_workdir = WORKDIR / f"point_{point_idx:03d}"
    param_toml_path_pt = ensure_param_file(param_dict, param_file=str(base_param_file), DIR=pt_workdir)
    print(f"Selected parameters (index {point_idx}): {dict(zip(parameter_names, selected_vector))}")
    print(f"Wrote parameter file: {param_toml_path_pt}")

    label_pt = f"point_{point_idx:03d}"
    selected = evaluate_water_properties(param_toml_path_pt, label_pt)

    if PLOT_PARITY:
        print(f"[{label_pt}] Computing parity-plot predictions...")
        if type_data == "pbc":
            E_sel_full, F_sel_full, Xi_sel_full = pbc_predict(param_toml_path_pt, label_pt, PARITY_NEEDED_IDX)
            _, E_sel_res = pbc_residualize(E_sel_full)
        else:
            E_sel_res, F_sel_full = cluster_predict(param_toml_path_pt, selected["e_water_relaxed_ha"], label_pt, only_idx=PARITY_NEEDED_IDX)
            Xi_sel_full = None
        parity_plots_for_point(
            label_pt, PARITY_DIR / label_pt,
            E_ref_res, F_ref_full, Xi_ref_full,
            E_base_res, F_base_full, Xi_base_full,
            E_sel_res, F_sel_full, Xi_sel_full,
            train_idx, test_idx, HAS_VIRIAL, PARITY_SPLITS,
        )

    # --- save results ---
    results = {
        "source_run_id": SOURCE_RUN_ID,
        "point_index": point_idx,
        "parameter_names": parameter_names,
        "parameters": {n: float(v) for n, v in zip(parameter_names, selected_vector)},
        "selected": selected,
        "baseline_default_gfn2": baseline,
        "hexamer_reference_kcal_mol": REF_BINDING_ENERGIES_KCAL,
        "reference_ordering": ref_order,
        "dft_reference_revpbe0_augccpvtz": {
            "monomer": DFT_MONOMER,
            "hexamers": DFT_HEXAMERS,
        },
    }
    (RUN_OUTPUT_DIR / f"results_idx_{point_idx:03d}.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(isomers))
    width = 0.2
    ax.bar(x - 1.5 * width, [selected["hexamer_binding_kcal_mol"][n] for n in isomers], width, label="GFN2-xTB (selected point)")
    ax.bar(x - 0.5 * width, [baseline["hexamer_binding_kcal_mol"][n] for n in isomers], width, color="orange", hatch="//", edgecolor="black", linewidth=0.8, label="GFN2-xTB (default baseline)")
    ax.bar(x + 0.5 * width, [DFT_HEXAMERS["hexamer_binding_kcal_mol"][n] for n in isomers], width, color="green", label="DFT (revPBE0/aug-cc-pVTZ)")
    ax.bar(x + 1.5 * width, [REF_BINDING_ENERGIES_KCAL[n] for n in isomers], width, label="CCSD(T)/AVDZ")
    ax.set_xticks(x)
    ax.set_xticklabels(isomers)
    ax.set_ylabel("Binding energy (kcal/mol)")
    ax.set_title(f"Water hexamers - idx {point_idx}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN_OUTPUT_DIR / f"hexamer_binding_idx_{point_idx:03d}.png", dpi=150)
    plt.close(fig)

print(f"\nWrote results to {RUN_OUTPUT_DIR}")
