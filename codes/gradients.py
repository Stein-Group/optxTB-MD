import numpy as np
from scipy.optimize import minimize
import re
import os
import subprocess
import matplotlib.pyplot as plt
import toml
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs, get_connectivity_matrix
from scipy import sparse
from ase.data import atomic_numbers as ZMAP
from pathlib import Path
import json

# import functions from another py file
from .calculate_ae_charged import write_custom_toml, generate_parameter_info, vector_to_param_changes
from .process_qchem_charged import parse_standard_nuclear_orientation

def _dftb_subprocess_env():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = os.environ.get("DFTB_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    # These are independent single-process calculations; avoid pinning every child
    # to an identical OpenMP place unless the caller explicitly opts in.
    env["OMP_PROC_BIND"] = os.environ.get("DFTB_OMP_PROC_BIND", "false")
    if "DFTB_OMP_PLACES" in os.environ:
        env["OMP_PLACES"] = os.environ["DFTB_OMP_PLACES"]
    else:
        env.pop("OMP_PLACES", None)
    return env

def calculate_dftbplus_energy(param_changes, coord_file, charge=0, DIR=None, param_path=None):
    # Ensure parameter file exists (cached if unchanged)
    DIR = Path(DIR) if DIR is not None else WORKDIR
    DIR.mkdir(parents=True, exist_ok=True)
    if param_path is None:
        param_path = ensure_param_file(param_changes, DIR=DIR)
    else:
        param_path = Path(param_path)

    # Write input in WORKDIR (no chdir)
    dftb_in_content = f'''Geometry = xyzFormat {{
    <<< "{coord_file}"
    }}

    Hamiltonian = xTB {{
      MaxSCCIterations = 2000
      SCCTolerance = 1e-10
      ParameterFile = "{param_path}"
      Charge = {charge}
      Mixer = DIIS {{
        Generations = 6
        UseFromStart = Yes
      }}
    }}

    Analysis = {{
      PrintForces = Yes
      MullikenAnalysis = Yes
      CM5 = {{}}
    }}

    Options {{
      WriteDetailedOut = Yes
      WriteCharges = Yes
    }}
    '''
    (DIR / "dftb_in.hsd").write_text(dftb_in_content)

    # Clean previous outputs
    for fn in ("detailed.out", "dftb.out"):
        p = DIR / fn
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # Prevent oversubscription (big win on multi-core machines)
    env = _dftb_subprocess_env()

    # Run DFTB+ in WORKDIR, stream stdout to file, don't hold stderr in memory
    output_file = DIR / "dftb.out"
    try:
        with output_file.open("w") as outfile:
            subprocess.run(
                ["dftb+"],
                check=True,
                cwd=str(DIR),
                stdout=outfile,
                stderr=subprocess.DEVNULL,
                env=env,
            )
    except subprocess.CalledProcessError as e:
        #print(f"Error running DFTB+: returncode={e.returncode}")
        # print last line from dftb.out for debugging
        print(open(output_file).read().splitlines()[-1])
        return None

    # Stream-parse energy (no full-file read)
    energy = None
    with output_file.open("r") as f:
        for line in f:
            m = ENERGY_RE.search(line)
            if m:
                energy = float(m.group(1))
                break
    return energy

def calculate_dftbplus_energy_reduced(param_changes, coord_file, DIR, charge=0):
    # Ensure parameter file exists (cached if unchanged)
    param_path = ensure_param_file(param_changes, DIR=DIR)

    # Write input in WORKDIR (no chdir)
    dftb_in_content = f'''Geometry = xyzFormat {{
    <<< "{coord_file}"
    }}

    Hamiltonian = xTB {{
      MaxSCCIterations = 2000
      SCCTolerance = 1e-10
      ParameterFile = "{param_path.name}"
      Charge = {charge}
      Mixer = DIIS {{
        Generations = 6
        UseFromStart = Yes
      }}
    }}

    Analysis = {{
      PrintForces = No
      MullikenAnalysis = No
    }}

    Options {{
      WriteDetailedOut = No
      WriteCharges = No
    }}
    '''
    (DIR / "dftb_in.hsd").write_text(dftb_in_content)

    # Clean previous outputs
    for fn in ("detailed.out", "dftb.out"):
        p = DIR / fn
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # Prevent oversubscription (big win on multi-core machines)
    env = _dftb_subprocess_env()

    # Run DFTB+ in DIR, stream stdout to file, don't hold stderr in memory
    output_file = DIR / "dftb.out"
    try:
        with output_file.open("w") as outfile:
            subprocess.run(
                ["dftb+"],
                check=True,
                cwd=str(DIR),
                stdout=outfile,
                stderr=subprocess.DEVNULL,
                env=env,
            )
    except subprocess.CalledProcessError as e:
        print(f"Error running DFTB+: returncode={e.returncode}")
        return None

    # Stream-parse energy (no full-file read)
    energy = None
    with output_file.open("r") as f:
        for line in f:
            m = ENERGY_RE.search(line)
            if m:
                energy = float(m.group(1))
                break
    return energy

def optimize_dftbplus_geometry_reduced(param_changes, coord_file, DIR, charge=0):
    # Ensure parameter file exists (cached if unchanged)
    param_path = ensure_param_file(param_changes, DIR=DIR)

    # Write input in WORKDIR (no chdir)
    dftb_in_content = f'''Geometry = xyzFormat {{
    <<< "{coord_file}"
    }}

    Driver = LBFGS {{
            MovedAtoms   = 1:-1
            MaxSteps     = 1000
            OutputPrefix = "geom.out"
        }}

    Hamiltonian = xTB {{
      MaxSCCIterations = 2000
      SCCTolerance = 1e-10
      ParameterFile = "{param_path.name}"
      Charge = {charge}
      Mixer = DIIS {{
        Generations = 6
        UseFromStart = Yes
      }}
    }}

    Analysis = {{
      PrintForces = No
      MullikenAnalysis = No
    }}

    Options {{
      WriteDetailedOut = No
      WriteCharges = No
    }}
    '''
    (DIR / "dftb_in.hsd").write_text(dftb_in_content)

    # Clean previous outputs
    for fn in ("detailed.out", "dftb.out"):
        p = DIR / fn
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # Prevent oversubscription (big win on multi-core machines)
    env = _dftb_subprocess_env()

    # Run DFTB+ in DIR, stream stdout to file, don't hold stderr in memory
    output_file = DIR / "dftb.out"
    try:
        with output_file.open("w") as outfile:
            subprocess.run(
                ["dftb+"],
                check=True,
                cwd=str(DIR),
                stdout=outfile,
                stderr=subprocess.DEVNULL,
                env=env,
            )
    except subprocess.CalledProcessError as e:
        print(f"Error running DFTB+: returncode={e.returncode}")
        return None

    # Stream-parse energy (no full-file read)
    energy = None
    with output_file.open("r") as f:
        #for line in f:
        for line in reversed(f.readlines()):
            m = ENERGY_RE.search(line)
            if m:
                energy = float(m.group(1))
                break
    return energy
        






# --------------------------------------------- dipole calculation ---------------------------------------------
# fallback working directory; the driver scripts always pass their own run directories
WORKDIR = Path(os.environ.get("SCRATCH", "/tmp")) / "optxTB-MD" / "work"
ENERGY_RE = re.compile(r"Total Energy:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*H")
FORCE_LINE_RE = re.compile(r"^\s*\d+\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$")

# cache last parameter hash to avoid redundant calculations
LAST_PARAM_HASH = {"h": None}

_REPO_ROOT = Path(__file__).resolve().parents[1]

def get_base_param_file() -> Path:
    try:
        out = subprocess.check_output(["tblite", "--version"], stderr=subprocess.STDOUT, text=True)
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
        ver = tuple(int(x) for x in m.groups()) if m else (0, 5, 0)
    except Exception:
        ver = (0, 5, 0)
    if ver >= (0, 6, 0):
        return _REPO_ROOT / "gfn2-xtb-060.toml"
    return _REPO_ROOT / "gfn2-xtb.toml"

def hash_params(param_changes) -> str:
    # robust hash independent of key order
    return json.dumps(param_changes, sort_keys=True, separators=(",", ":"))

def ensure_param_file(param_changes, param_file=None, DIR=WORKDIR):
    """Write new_parameters.toml only if param_changes changed."""
    if param_file is None:
        param_file = get_base_param_file()
    #print(f'Current dir: {DIR}')
    DIR = Path(DIR)
    DIR.mkdir(parents=True, exist_ok=True)
    h = hash_params(param_changes)
    out_path = DIR / "new_parameters.toml"
    #print(f"Parameter hash: {h}")
    #print(f"Searching for parameter file at: {out_path}")
    #print(out_path.exists())
    #if LAST_PARAM_HASH["h"] == h and out_path.exists():
    #    return out_path  # already up-to-date
    if out_path.exists():
        out_path.unlink()
    # (Re)build from base toml
    import toml
    base = toml.load(param_file)
    for element, changes in param_changes.items():
        if element in base.get('element', {}):
            for param, value in changes.items():
                if param in base['element'][element]:
                    base['element'][element][param] = value
        elif element in base.get('hamiltonian', {}).get('xtb', {}):
            for param, value in changes.items():
                if param in base.get('hamiltonian', {}).get('xtb', {}).get('kpair', {}):
                    #print('kpair', element, param, value)
                    base['hamiltonian']['xtb'][element][param] = value
    # your helper name stayed the same:
    write_custom_toml(base, str(out_path))
    LAST_PARAM_HASH["h"] = h
    return out_path

def parse_forces_from_detailed(det_path: Path):
    """Fast one-pass extraction of the last 'Total Forces' block."""
    forces = []
    grab = False
    with det_path.open("r") as f:
        for line in f:
            ls = line.strip()
            if ls.startswith("Total Forces"):
                grab = True
                forces = []  # reset to last block seen
                continue
            if grab:
                if not ls:  # blank line ends the block
                    grab = False
                    continue
                m = FORCE_LINE_RE.match(ls)
                if m:
                    fx, fy, fz = map(float, m.groups())
                    forces.append([fx, fy, fz])
                else:
                    # unexpected line = end of block
                    grab = False
    return forces

molecule_formulas = {
    'H2O': {'H': 2, 'O': 1},
    'Na': {'Na': 1},
    'Cl': {'Cl': 1}
}

def get_molecule_formula(molecule):
    atom_counts = {}
    for atom in molecule.get_chemical_symbols():
        atom_counts[atom] = atom_counts.get(atom, 0) + 1
    return atom_counts
    
def check_molecule(molecule, formulas):
    formula = get_molecule_formula(molecule)
    for name, ref_formula in formulas.items():
        if formula == ref_formula:
            return name, True
    return None, False


DEBYE_PER_AU = 1 / 0.393430238326893
BOHR_PER_ANG = 1.88972598857892

def parse_charges(det_path, use_cm5=False):
    lines = Path(det_path).read_text().splitlines()
    label = "CM5 corrected atomic gross charges (e)" if use_cm5 else "Atomic gross charges (e)"
    try:
        start = next(i for i, ln in enumerate(lines) if label in ln)
    except StopIteration:
        raise RuntimeError(f"Block '{label}' not found in {det_path}")
    i = start + 2
    charges = []
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"\s*(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$", ln)
        if not m:
            break
        charges.append(float(m.group(2)))
        i += 1
    return np.array(charges, dtype=float)

def read_xyz(xyz_path):
    with open(xyz_path, "r") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    header = lines[1].strip() if len(lines) > 1 else ""
    species, coords = [], []
    for ln in lines[2:2+n]:
        parts = ln.split()
        if len(parts) < 4:
            raise RuntimeError("Malformed XYZ line:\n" + ln)
        species.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords, dtype=float), species, header

def compute_dipole(coords, charges, idx_1based=None, origin=None):
    if idx_1based is None:
        mask = np.arange(len(charges))
    else:
        mask = np.array(idx_1based, dtype=int) - 1  # convert to 0-based
        #print(f"Mask = {mask}")
    R = coords[mask].copy()
    # (3,) -> (1, 3) is done via [None, :] (additional dimension)
    if origin is not None:
        R -= origin[None, :]
    q = charges[mask]
    #print(f"R={R}, origin={origin}, q={q}")
    mu_eA = np.sum(q[:, None] * R, axis=0)   # e·Å
    mu_au = mu_eA * BOHR_PER_ANG            # e·Bohr
    mu_D  = mu_au * DEBYE_PER_AU            # Debye
    return mu_eA, mu_au, mu_D, q.sum()

def split_components_indices(structure, scale=1.0):
    cutoff = natural_cutoffs(structure, mult=scale, Na=0.8)
    nl = NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(structure)
    connectivity = nl.get_connectivity_matrix()
    n_comp, comp_labels = sparse.csgraph.connected_components(connectivity)
    return [np.where(comp_labels == i)[0] for i in range(n_comp)]

def center_of_nuclear_charges(coords, species):
    Z = np.array([ZMAP[s] for s in species], dtype=float)
    return (Z[:, None] * coords).sum(axis=0) / Z.sum()

def calculate_charges_dipole(structure, species, coords, charges, verbose=True):
    comp_indices = split_components_indices(structure, scale=1.0)
    results = []
    muDs = []
    water_count = 0
    for idx in comp_indices:
        mol_atoms = structure[idx]
        name, ok = check_molecule(mol_atoms, molecule_formulas)
        if not ok or name != "H2O":
            continue
        water_count += 1
        species_i = [species[j] for j in idx]
        coords_i  = coords[idx]
        r0_i = center_of_nuclear_charges(coords_i, species_i)
        idx_1based = (idx + 1).tolist()
        _, _, mu_D_i, qsum_i = compute_dipole(coords, charges, idx_1based=idx_1based, origin=r0_i)
        #print(f'Components = {mu_D_i}')
        muDs.append(np.linalg.norm(mu_D_i))
        results.append({
            "mol_id": water_count,
            "indices_1based": idx_1based,
            "origin_nuclear_charge_center": r0_i,
            "net_charge_e": qsum_i,
            "mu_D": mu_D_i
        })
    if not results:
        if verbose:
            print("No H2O molecules detected. Try adjusting 'scale' or check connectivity.")
        return results, np.array([])
    return results, np.array(muDs)

def calculate_e_unbiased_f_md(param_changes, structure_file, chrg=0):
    # Run DFTB+ (this also ensures parameters exist via cache inside)
    energy_self = calculate_dftbplus_energy(param_changes, structure_file, chrg)
    det_path = WORKDIR / "detailed.out"

    # Fast parsing of forces and CM5 charges from the same file
    forces = parse_forces_from_detailed(det_path)
    charges = parse_charges(det_path, use_cm5=True)

    # Read coords/species once and reuse for both ASE and numpy
    coords, species, _ = read_xyz(structure_file)
    if len(charges) != len(coords):
        raise RuntimeError(f"#atoms mismatch: charges={len(charges)} vs xyz={len(coords)}")

    # Build ASE object directly from arrays (avoids a second disk read)
    structure_ase = Atoms(symbols=species, positions=coords)
    structure_ase.set_pbc(False)

    _, muDs = calculate_charges_dipole(structure_ase, species, coords, charges)
    return energy_self, forces, muDs

def get_total_charge(structure_file):
    # Avoid ASE I/O overhead: reuse your fast XYZ reader
    _, species, _ = read_xyz(structure_file)
    # Na = +1, Cl = -1, others = 0
    return int(sum(1 if s == "Na" else (-1 if s == "Cl" else 0) for s in species))

def get_number_of_na_cl_waters(structure_file):
    _, species, _ = read_xyz(structure_file)
    return species.count('Na'), species.count('Cl'), species.count('O')

def calculate_re_f_md(param_changes, structure_file, chrg=0, run_dir=None, monomer_energies=None, monomer_xyz_root=None, param_path=None, verbose=True):
    run_dir = Path(run_dir) if run_dir is not None else WORKDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    # Run DFTB+ (this also ensures parameters exist via cache inside)
    energy_self = calculate_dftbplus_energy(param_changes, structure_file, chrg, DIR=run_dir, param_path=param_path)
    #print('Calculated E_self')
    number_of_na, number_of_cl, number_of_waters = get_number_of_na_cl_waters(structure_file)
    if monomer_energies is not None:
        e_na, e_cl, e_water = monomer_energies
    else:
        monomer_xyz_root = Path(monomer_xyz_root) if monomer_xyz_root is not None else WORKDIR / 'monomers_xtb'
        monomer_run_root = run_dir / 'monomers_xtb'
        e_na, e_cl, e_water = 0.0, 0.0, 0.0
        if number_of_na != 0:
            e_na = calculate_dftbplus_energy_reduced(param_changes, monomer_xyz_root/'na/na.xyz', monomer_run_root/'na', charge=1)
        #print('Calculated E_Na')
        if number_of_cl != 0:
            e_cl = calculate_dftbplus_energy_reduced(param_changes, monomer_xyz_root/'cl/cl.xyz', monomer_run_root/'cl', charge=-1)
        if number_of_waters != 0:
            e_water = optimize_dftbplus_geometry_reduced(param_changes, monomer_xyz_root/'water/water.xyz', monomer_run_root/'water', charge=0)
        #e_water = calculate_dftbplus_energy_reduced(param_changes, monomer_xyz_root/'water/water.xyz', monomer_run_root/'water', charge=0)
    re = energy_self - number_of_na * e_na - number_of_cl * e_cl - number_of_waters * e_water
    #print(f'For a structure {structure_file}, energies are E_self={energy_self}, E_Na={e_na}, E_Cl={e_cl}, E_water={e_water}. Thus, RE={re}.')
    det_path = run_dir / "detailed.out"
    # Fast parsing of forces and CM5 charges from the same file
    forces = parse_forces_from_detailed(det_path)
    charges = parse_charges(det_path, use_cm5=True)

    # Read coords/species once and reuse for both ASE and numpy
    coords, species, _ = read_xyz(structure_file)
    if len(charges) != len(coords):
        raise RuntimeError(f"#atoms mismatch: charges={len(charges)} vs xyz={len(coords)}")

    # Build ASE object directly from arrays (avoids a second disk read)
    structure_ase = Atoms(symbols=species, positions=coords)
    structure_ase.set_pbc(False)

    _, muDs = calculate_charges_dipole(structure_ase, species, coords, charges, verbose=verbose)
    return re, forces, muDs
