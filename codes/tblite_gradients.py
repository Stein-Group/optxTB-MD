from pathlib import Path
import numpy as np
from ase import Atoms
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from tqdm.auto import tqdm
import subprocess
import time

from .molden_handling import run_tblite_cli, parse_tblite_tag_energy_gradient, MoldenTemplate, MoldenTemplateCache, gradient_to_forces, write_molden_from_tblite
from .gradients import ensure_param_file, read_xyz, get_total_charge, calculate_charges_dipole
from .multiwfn_wrapper import run_multiwfn_charges

ENERGY_RE = re.compile(r"Total Energy:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*H")

def calculate_dftbplus_energy_reduced(param_path, coord_file, DIR, charge=0):
    # Ensure parameter file exists (cached if unchanged)
    #param_path = ensure_param_file(param_changes, DIR=DIR)
    DIR = Path(DIR)
    DIR.mkdir(parents=True, exist_ok=True)

    # Write input in WORKDIR (no chdir)
    dftb_in_content = f'''Geometry = xyzFormat {{
    <<< "{coord_file}"
    }}

    Hamiltonian = xTB {{
      MaxSCCIterations = 2000
      SCCTolerance = 1e-10
      ParameterFile = "{str(param_path)}"
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
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        raise RuntimeError(f"DFTB+ monomer failed: returncode={e.returncode}") from e
        #print(f"Error running DFTB+: returncode={e.returncode}")
        #return None

    # Stream-parse energy (no full-file read)
    energy = None
    with output_file.open("r") as f:
        for line in f:
            m = ENERGY_RE.search(line)
            if m:
                energy = float(m.group(1))
                break
    return energy

def optimize_dftbplus_geometry_reduced(param_path, coord_file, DIR, charge=0):
    # Ensure parameter file exists (cached if unchanged)
    #param_path = ensure_param_file(param_changes, DIR=DIR)
    DIR = Path(DIR)
    DIR.mkdir(parents=True, exist_ok=True)

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
      ParameterFile = "{param_path}"
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
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        raise RuntimeError(f"DFTB+ monomer failed: returncode={e.returncode}") from e
        #print(f"Error running DFTB+: returncode={e.returncode}")
        #return None

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

def count_na_cl_waters_from_xyz(xyz_path: str | Path):
    coords, species, _ = read_xyz(str(xyz_path))
    n_na = sum(s == "Na" for s in species)
    n_cl = sum(s == "Cl" for s in species)
    n_o  = sum(s == "O"  for s in species)
    n_h  = sum(s == "H"  for s in species)

    # обычно вода = число кислородов
    n_w = n_o
    # мягкая самопроверка
    if n_h // 2 != n_w:
        # не падаем, просто используем n_o
        pass
    return n_na, n_cl, n_w



def compute_monomer_energies_dftbplus(
    *,
    param_toml_path: Path,
    monomer_na_xyz: Path,
    monomer_cl_xyz: Path,
    monomer_wat_xyz: Path,
    dftb_run_root: Path,
    ):
    e_na = calculate_dftbplus_energy_reduced(
        param_path=param_toml_path,
        coord_file=monomer_na_xyz,
        DIR=dftb_run_root / "na",
        charge=1,
    )
    e_cl = calculate_dftbplus_energy_reduced(
        param_path=param_toml_path,
        coord_file=monomer_cl_xyz,
        DIR=dftb_run_root / "cl",
        charge=-1,
    )
    e_w = optimize_dftbplus_geometry_reduced(
        param_path=param_toml_path,
        coord_file=monomer_wat_xyz,
        DIR=dftb_run_root / "water",
        charge=0,
    )
    return e_na, e_cl, e_w


def analyze_one_structure_tblite_multiwfn_charges(
    *,
    xyz_path: Path,
    total_charge: int,
    param_toml_path: Path,
    template_cache,
    tblite_root: Path,
    molden_out_dir: Path,
    multiwfn_exe: str,
    atmrad_dir: Path,
    multiwfn_root: Path,
    scheme: str = "CM5",
    iterations: int = 2000,
    restart_policy: str = "recreate",
    multiwfn_timeout_s: int = 1800,
    omp_tblite: int = 1,
    omp_multiwfn: int = 1,
    monomer_energies: tuple[float,float,float] | None = None,  # (e_na,e_cl,e_w)
    param_changes: dict | None = None,
):
    stem = xyz_path.stem

    # ---- 1) tblite run dir (уникальный)
    run_dir = Path(tblite_root) / f"{stem}__q{total_charge:+d}"
    coords, species, _ = read_xyz(str(xyz_path))
    nat = len(coords)

    # tblite run -> tag/json/restart/out
    tag_path, json_path, restart_path, out_path = run_tblite_cli(
        xyz_path=xyz_path,
        run_dir=run_dir,
        param_toml_path=param_toml_path,
        charge=total_charge,
        iterations=iterations,
        restart_policy=restart_policy,
        omp_tblite = omp_tblite,
    )

    # ---- 2) energy + forces из tag
    E, grad = parse_tblite_tag_energy_gradient(tag_path, nat=nat)
    F = gradient_to_forces(grad)
    #print(f"E={E}")

    # ---- 3) molden reconstruction (по шаблону)
    templ = template_cache.get(stem, total_charge)
    molden_out_dir = Path(molden_out_dir)
    molden_out_dir.mkdir(parents=True, exist_ok=True)
    molden_path = molden_out_dir / f"{stem}__q{total_charge:+d}.molden"
    write_molden_from_tblite(
        templ,
        restart_path=restart_path,
        out_path=molden_path,
        param_toml_path=param_toml_path,
        param_changes=param_changes,
        add_spherical_tags=True,
        valence_map=None,
    )

    # ---- 4) Multiwfn charges (scheme)
    mw_run_dir = Path(multiwfn_root) / molden_path.stem
    q = run_multiwfn_charges(
        molden_path,
        multiwfn_exe=multiwfn_exe,
        timeout_s=multiwfn_timeout_s,
        omp_threads=omp_multiwfn,
        atmrad_dir=atmrad_dir,
        workdir=mw_run_dir,
        scheme=scheme,
        out_chg_path=None,
    )

    if len(q) != nat:
        raise RuntimeError(f"{xyz_path.name}: charges length {len(q)} != nat {nat}")

    # ---- 5) Relative energy (если дали monomer energies)
    re = float(E)
    if monomer_energies is not None:
        e_na, e_cl, e_w = monomer_energies
        n_na, n_cl, n_w = count_na_cl_waters_from_xyz(xyz_path)
        re = float(E) - n_na*e_na - n_cl*e_cl - n_w*e_w

    return float(re), np.array(F), np.array(q), molden_path





def analyze_one_structure_tblite(
    *,
    xyz_path: Path,
    total_charge: int,
    param_toml_path: Path,
    tblite_root: Path,
    iterations: int = 2000,
    restart_policy: str = "recreate",
    omp_tblite: int = 1,
    monomer_energies: tuple[float,float,float] | None = None,  # (e_na,e_cl,e_w)
):
    """Same as analyze_one_structure_tblite_multiwfn_charges, but skips the
    Molden write + Multiwfn step entirely (no partial charges). Cheap E/F-only path."""
    stem = xyz_path.stem

    # ---- 1) tblite run dir (уникальный)
    run_dir = Path(tblite_root) / f"{stem}__q{total_charge:+d}"
    coords, species, _ = read_xyz(str(xyz_path))
    nat = len(coords)

    # tblite run -> tag/json/restart/out
    tag_path, json_path, restart_path, out_path = run_tblite_cli(
        xyz_path=xyz_path,
        run_dir=run_dir,
        param_toml_path=param_toml_path,
        charge=total_charge,
        iterations=iterations,
        restart_policy=restart_policy,
        omp_tblite = omp_tblite,
    )

    # ---- 2) energy + forces из tag
    E, grad = parse_tblite_tag_energy_gradient(tag_path, nat=nat)
    F = gradient_to_forces(grad)

    # ---- 3) Relative energy (если дали monomer energies)
    re = float(E)
    if monomer_energies is not None:
        e_na, e_cl, e_w = monomer_energies
        n_na, n_cl, n_w = count_na_cl_waters_from_xyz(xyz_path)
        re = float(E) - n_na*e_na - n_cl*e_cl - n_w*e_w

    return float(re), np.array(F)

def calculate_dict_tblite_parallel(
    *,
    param_changes,
    coord_files,
    dataset_path: str | Path,
    tblite_root: Path,
    iterations: int = 2000,
    restart_policy: str = "recreate",
    max_workers: int | None = None,
    monomer_na_xyz: Path | None = None,
    monomer_cl_xyz: Path | None = None,
    monomer_wat_xyz: Path | None = None,
    MONO_DIR: Path | None = None,
    compute_relative_energy: bool = True,
    WORKDIR: str | Path,
    show_progress: bool = True,
    print_timing: bool = False,
):
    """Same as calculate_dict_tblite_multiwfn_parallel_charges, but skips the
    Molden + Multiwfn step entirely (no partial charges). Use when charges are
    not needed -- this is the cheap E/F-only path, no Multiwfn invocation."""
    dataset_path = Path(dataset_path)
    tblite_root = Path(tblite_root)

    # create toml omce
    param_toml_path = ensure_param_file(param_changes, DIR=WORKDIR)

    # count monomers once for a given parameter vector
    monomer_energies = None
    if compute_relative_energy:
        monomer_start = time.time()
        if monomer_na_xyz is None or monomer_cl_xyz is None or monomer_wat_xyz is None:
            raise RuntimeError("Need monomer_na_xyz/monomer_cl_xyz/monomer_wat_xyz to compute relative energy.")
        if MONO_DIR is None:
            MONO_DIR = Path(monomer_na_xyz).parent
        monomer_energies = compute_monomer_energies_dftbplus(
            param_toml_path=param_toml_path,
            monomer_na_xyz=Path(monomer_na_xyz),
            monomer_cl_xyz=Path(monomer_cl_xyz),
            monomer_wat_xyz=Path(monomer_wat_xyz),
            dftb_run_root=Path(MONO_DIR),
        )
        if print_timing:
            print(f"Monomer energy setup time: {time.time() - monomer_start:.2f} seconds")

    ncores = os.cpu_count() or 1
    if max_workers is None:
        max_workers = min(ncores, len(coord_files))

    # tqdm
    if show_progress:
        pbar = tqdm(total=len(coord_files), desc="tblite (E/F only)", unit="file")
    else:
        pbar = None

    # this is parallelized - multiple tblite runs simultaneously
    def job(i_filename):
        i, filename = i_filename
        xyz_path = dataset_path / filename
        total_charge = get_total_charge(xyz_path)

        re, F = analyze_one_structure_tblite(
            xyz_path=xyz_path,
            total_charge=total_charge,
            param_toml_path=param_toml_path,
            tblite_root=tblite_root,
            iterations=iterations,
            restart_policy=restart_policy,
            monomer_energies=monomer_energies,
        )
        return i, re, F

    # saving order as in coord_files for parallelization
    idx_files = list(enumerate([fn for _, fn in coord_files]))
    res   = [None] * len(idx_files)
    forces = [None] * len(idx_files)

    errors = {}

    batch_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(job, x) for x in idx_files]
        for fut in as_completed(futs):
            try:
                i, re_i, F_i = fut.result()
                res[i] = re_i
                forces[i] = F_i
            except Exception as e:
                # save error, but do not stop the calculations (?)
                errors[str(fut)] = repr(e)
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()
    if print_timing:
        print(f"tblite-only batch time ({len(coord_files)} structures, {max_workers} workers): {time.time() - batch_start:.2f} seconds")

    failed = [
        i for i in range(len(res))
        if res[i] is None or forces[i] is None
    ]
    if failed:
        raise RuntimeError(
            f"Some jobs failed: {len(failed)}/{len(res)}. "
            f"Example error: {next(iter(errors.values())) if errors else 'unknown'}"
        )

    return np.array(res, dtype=float), forces

def calculate_dict_tblite_multiwfn_parallel_charges(
    *,
    param_changes,
    coord_files,
    dataset_path: str | Path,
    template_cache,
    tblite_root: Path,
    molden_out_dir: Path,
    multiwfn_root: Path,
    multiwfn_exe: str,
    atmrad_dir: Path,
    scheme: str = "CM5",
    iterations: int = 2000,
    restart_policy: str = "recreate",
    max_workers: int | None = None,
    omp_multiwfn: int = 1,
    monomer_na_xyz: Path | None = None,
    monomer_cl_xyz: Path | None = None,
    monomer_wat_xyz: Path | None = None,
    MONO_DIR: Path | None = None,
    compute_relative_energy: bool = True,
    clean_multiwfn_root: bool = True,
    WORKDIR: str | Path,
    show_progress: bool = True,
    print_timing: bool = False,
):
    dataset_path = Path(dataset_path)
    tblite_root = Path(tblite_root)
    molden_out_dir = Path(molden_out_dir)
    multiwfn_root = Path(multiwfn_root)
    atmrad_dir = Path(atmrad_dir)

    # create toml omce
    param_toml_path = ensure_param_file(param_changes, DIR=WORKDIR)

    # count monomers once for a given parameter vector
    monomer_energies = None
    if compute_relative_energy:
        monomer_start = time.time()
        if monomer_na_xyz is None or monomer_cl_xyz is None or monomer_wat_xyz is None:
            raise RuntimeError("Need monomer_na_xyz/monomer_cl_xyz/monomer_wat_xyz to compute relative energy.")
        if MONO_DIR is None:
            MONO_DIR = Path(monomer_na_xyz).parent
        #monomer_energies = compute_monomer_energies_tblite(
        #    param_toml_path=param_toml_path,
        #    monomer_na_xyz=Path(monomer_na_xyz),
        #    monomer_cl_xyz=Path(monomer_cl_xyz),
        #    monomer_wat_xyz=Path(monomer_wat_xyz),
        #    tblite_root=tblite_root,
        #    iterations=iterations,
        #    restart_policy="recreate",
        #)
        monomer_energies = compute_monomer_energies_dftbplus(
            param_toml_path=param_toml_path,
            monomer_na_xyz=Path(monomer_na_xyz),
            monomer_cl_xyz=Path(monomer_cl_xyz),
            monomer_wat_xyz=Path(monomer_wat_xyz),
            dftb_run_root=Path(MONO_DIR),
        )
        if print_timing:
            print(f"Monomer energy setup time: {time.time() - monomer_start:.2f} seconds")
        #print(monomer_energies)

    ncores = os.cpu_count() or 1
    if max_workers is None:
        max_workers = min(ncores, len(coord_files))

    # clean multiwfn_root 
    if clean_multiwfn_root and multiwfn_root.exists():
        shutil.rmtree(multiwfn_root, ignore_errors=True)
    multiwfn_root.mkdir(parents=True, exist_ok=True)

    # tqdm
    if show_progress:
        pbar = tqdm(total=len(coord_files), desc=f"tblite+multiwfn ({scheme})", unit="file")
    else:
        pbar = None

    # this is parallelized - multiple tblite runs simultaneously
    def job(i_filename):
        i, filename = i_filename
        xyz_path = dataset_path / filename
        total_charge = get_total_charge(xyz_path)

        re, F, q, molden_path = analyze_one_structure_tblite_multiwfn_charges(
            xyz_path=xyz_path,
            total_charge=total_charge,
            param_toml_path=param_toml_path,
            template_cache=template_cache,
            tblite_root=tblite_root,
            molden_out_dir=molden_out_dir,
            multiwfn_exe=multiwfn_exe,
            atmrad_dir=atmrad_dir,
            multiwfn_root=multiwfn_root,
            scheme=scheme,
            iterations=iterations,
            restart_policy=restart_policy,
            monomer_energies=monomer_energies,
            omp_multiwfn=omp_multiwfn,
            param_changes=param_changes,
        )
        return i, re, F, q

    # saving order as in coord_files for parallelization 
    idx_files = list(enumerate([fn for _, fn in coord_files]))
    res   = [None] * len(idx_files)
    forces = [None] * len(idx_files)
    chgs  = [None] * len(idx_files)

    errors = {}

    batch_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(job, x) for x in idx_files]
        for fut in as_completed(futs):
            try:
                i, re_i, F_i, q_i = fut.result()
                res[i] = re_i
                forces[i] = F_i
                chgs[i] = q_i
            except Exception as e:
                # save error, but do not stop the calculations (?)
                errors[str(fut)] = repr(e)
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()
    if print_timing:
        print(f"tblite+Multiwfn batch time ({len(coord_files)} structures, {max_workers} workers, {omp_multiwfn} Multiwfn threads): {time.time() - batch_start:.2f} seconds")

    failed = [
        i for i in range(len(res))
        if res[i] is None or forces[i] is None or chgs[i] is None
    ]
    if failed:
        raise RuntimeError(
            f"Some jobs failed: {len(failed)}/{len(res)}. "
            f"Example error: {next(iter(errors.values())) if errors else 'unknown'}"
        )
    #if any(x is None for x in res):
        # если были ошибки — лучше явно сказать
    #    raise RuntimeError(
    #        f"Some jobs failed: {sum(x is None for x in res)}/{len(res)}. "
    #        f"Example error: {next(iter(errors.values())) if errors else 'unknown'}"
    #    )

    return np.array(res, dtype=float), forces, chgs


