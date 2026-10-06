from pathlib import Path
import os, shutil, tempfile, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from ase import Atoms
import resource

from .multiwfn_wrapper import parse_multiwfn_chg, _ensure_atmrad, _check_multiwfn_output
from .process_qchem_charged import extract_final_energy, extract_forces
from .gradients import calculate_charges_dipole, read_xyz, get_number_of_na_cl_waters

def _link_or_copy(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    if src.resolve() == dst.resolve():
        return
    try:
        os.link(src, dst)  # hardlink
    except Exception:
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        except Exception:
            shutil.copyfile(src, dst)

def raise_stack_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    # hard может быть RLIM_INFINITY или большое число
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))

def _find_chg_file(run_dir: Path, wfn_path: Path) -> Path | None:
    candidates = [
        run_dir / f"{wfn_path.stem}.chg",
        run_dir / "input.chg",
        run_dir / "molden.chg",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: любой .chg, самый новый
    chgs = sorted(run_dir.glob("*.chg"), key=lambda p: p.stat().st_mtime)
    return chgs[-1] if chgs else None


def run_multiwfn_charges_wfnfile(
    wfn_path: str | Path,
    *,
    nat: int,
    scheme: str = "CM5",
    multiwfn_exe: str,
    atmrad_dir: str | Path | None = None,
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    timeout_s: int = 3800,
    omp_threads: int = 1,
    out_chg_path: str | Path | None = None,
    trsp_tol: float = 1e-3,
    fatal_problem_wfn: bool = True,
) -> np.ndarray:
    wfn_path = Path(wfn_path)

    match scheme:
        case "hirshfeldI":
            menu_item = "15"
        case "CM5":
            menu_item = "16"
        case "ADCH":
            menu_item = "11"
        case _:
            raise ValueError("scheme must be one of: hirshfeldI, CM5, ADCH")

    temp_ctx = None
    if workdir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"multiwfn_{scheme}_")
        run_dir = Path(temp_ctx.name)
    else:
        run_dir = Path(workdir)
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # сохраняем расширение входа, чтобы Multiwfn точно понял формат
        in_name = run_dir / f"input{wfn_path.suffix}"
        _link_or_copy(wfn_path, in_name)

        # чистим старые .chg, чтобы не было мусора
        for old in run_dir.glob("*.chg"):
            try: old.unlink()
            except Exception: pass

        # более устойчивый menu:
        # - после "1 Start calculation" даём "1" на выбор atomic densities (если не спросит — не страшно)
        # - потом "y" на запись .chg
        menu = "\n".join([
            "7",
            menu_item,
            "1",
            "y",   # <-- сохранить .chg
            "0",
            "q",
            ""
        ])

        env = os.environ.copy()

        # --- Железобетонно запретить Multiwfn >1 потока
        env["OMP_NUM_THREADS"] = str(omp_threads)
        #env["OMP_THREAD_LIMIT"] = "1"          # <- ключевое: даже если Multiwfn сам выставит 4, рантайм не даст
        #env["OMP_MAX_ACTIVE_LEVELS"] = "1"
        #env["OMP_DYNAMIC"] = "FALSE"

        # --- Стек для OMP (самая частая причина SIGSEGV у Multiwfn на больших системах)
        env["OMP_STACKSIZE"] = "4096M"         # начни с 2GB; если надо, попробуем 4096M

        # --- Intel OpenMP (forrtl часто реагирует лучше, если задать и это)
        env["KMP_STACKSIZE"] = "4096M"
        env["KMP_AFFINITY"] = "disabled"

        # --- Чтобы никакие BLAS/MKL не создавали свои потоки
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

        mw_bin_dir = Path(multiwfn_exe).resolve().parent
        settings = mw_bin_dir / "settings.ini"
        if settings.exists():
            # чтобы Multiwfn точно видел settings.ini
            shutil.copyfile(settings, Path(workdir) / "settings.ini")
            env["Multiwfnpath"] = str(mw_bin_dir)


        if atmrad_dir is not None:
            _ensure_atmrad(run_dir, atmrad_dir, mode="symlink")

        raise_stack_limit()

        proc = subprocess.run(
            [multiwfn_exe, str(in_name.name)],
            input=menu.encode("utf-8", errors="ignore"),
            cwd=str(run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            env=env,
            check=False,
        )

        #print("OMP_NUM_THREADS =", env.get("OMP_NUM_THREADS"))
        #print("OMP_THREAD_LIMIT =", env.get("OMP_THREAD_LIMIT"))
        #print("OMP_STACKSIZE =", env.get("OMP_STACKSIZE"))
        #print("Multiwfnpath =", env.get("Multiwfnpath"))
        #print("workdir =", workdir)


        stdout_text = proc.stdout.decode("utf-8", errors="ignore")
        _check_multiwfn_output(stdout_text, trsp_tol=trsp_tol, fatal_problem_wfn=fatal_problem_wfn)

        chg_file = _find_chg_file(run_dir, wfn_path)
        if chg_file is None:
            msg = stdout_text[-6000:]
            raise RuntimeError(f"Multiwfn produced no .chg for {wfn_path}\n--- last output ---\n{msg}\n")

        charges = parse_multiwfn_chg(chg_file, nat=nat)


        if out_chg_path is not None:
            out_chg_path = Path(out_chg_path)
            out_chg_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chg_file, out_chg_path)

        return charges

    finally:
        if temp_ctx is not None and not keep_workdir:
            temp_ctx.cleanup()


def precompute_multiwfn_charges_from_fchk(
    *,
    fchk_dir: str | Path,
    xyz_dir: str | Path,                 # чтобы брать nat + порядок атомов
    chg_root: str | Path,
    multiwfn_exe: str,
    atmrad_dir: str | Path | None = None,
    schemes: tuple[str, ...] = ("CM5", "hirshfeldI", "ADCH"),
    stems: set[str] | None = None,       # restrict to these xyz/fchk stems only
    max_workers: int | None = None,
    omp_multiwfn: int = 1,
    tmp_root: str | Path = "/tmp/multiwfn_ref_runs",
    overwrite: bool = False,
    show_progress: bool = True,
):
    fchk_dir = Path(fchk_dir)
    xyz_dir  = Path(xyz_dir)
    chg_root = Path(chg_root)
    tmp_root = Path(tmp_root)

    # reference charges already cached for every requested structure: the .fchk files are not needed
    if stems is not None and not overwrite and all(
        (chg_root / scheme / f"{stem}.chg").exists() for scheme in schemes for stem in stems
    ):
        print(f"All {len(stems)} reference charges found in {chg_root} for {schemes}; skipping Multiwfn on .fchk.")
        return chg_root

    fchk_paths = sorted(fchk_dir.glob("*.fchk"))
    if stems is not None:
        fchk_paths = [p for p in fchk_paths if p.stem in stems]
    if not fchk_paths:
        raise RuntimeError(f"No .fchk found in {fchk_dir} matching the requested stems.")

    ncores = os.cpu_count() or 1
    if max_workers is None:
        max_workers = min(ncores, len(fchk_paths))

    # быстрый маппинг stem -> xyz_path (ожидаем одинаковые stem)
    xyz_map = {p.stem: p for p in xyz_dir.glob("*.xyz")}

    for scheme in schemes:
        out_dir = chg_root / scheme
        out_dir.mkdir(parents=True, exist_ok=True)

        # tqdm на схему
        pbar = tqdm(total=len(fchk_paths), desc=f"Multiwfn {scheme}", unit="file") if show_progress else None

        def job(fchk_path: Path):
            stem = fchk_path.stem
            if stem not in xyz_map:
                raise RuntimeError(f"Can't find matching xyz for {stem} in {xyz_dir}")
            xyz_path = xyz_map[stem]

            # nat из xyz
            coords, species, _ = read_xyz(str(xyz_path))
            nat = len(coords)

            out_chg = out_dir / f"{stem}.chg"
            if out_chg.exists() and not overwrite:
                return stem, out_chg  # skip

            # уникальная временная папка под этот файл
            wdir = tmp_root / scheme / stem
            wdir.mkdir(parents=True, exist_ok=True)

            q = run_multiwfn_charges_wfnfile(
                fchk_path,
                nat=nat,
                scheme=scheme,
                multiwfn_exe=multiwfn_exe,
                atmrad_dir=atmrad_dir,
                workdir=wdir,
                omp_threads=omp_multiwfn,
                out_chg_path=out_chg,
            )
            # sanity: сумма зарядов ≈ total_charge из xyz (если хочешь)
            return stem, out_chg

        errors = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_stem = {}
            for fchk_path in fchk_paths:
                fut = ex.submit(job, fchk_path)
                future_to_stem[fut] = fchk_path.stem

            for fut in as_completed(list(future_to_stem.keys())):
                stem = future_to_stem[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors[stem] = repr(e)
                finally:
                    if pbar is not None:
                        pbar.update(1)

        if pbar is not None:
            pbar.close()

        if errors:
            # покажем первые несколько
            first = next(iter(errors.items()))
            raise RuntimeError(f"Scheme {scheme}: {len(errors)} jobs failed. First: {first[0]} -> {first[1]}")

    return chg_root

def calculate_re_f_md_dft_with_precomputed_chg(
    structure_file: str | Path,
    *,
    dft_out_dir: str | Path,          # где лежат *.out
    chg_root: str | Path,             # <chg_root>/<scheme>/<stem>.chg
    scheme: str,                      # "CM5" | "hirshfeldI" | "ADCH"
    monomer_out_dir: str | Path,      # где na.out, cl.out, water.out
):
    structure_file = Path(structure_file)
    stem = structure_file.stem

    dft_out_dir = Path(dft_out_dir)
    out_file = dft_out_dir / f"{stem}.out"
    if not out_file.exists():
        raise RuntimeError(f"Missing DFT out: {out_file}")

    # энергии/forces — как раньше
    energy_self = extract_final_energy(str(out_file))
    forces = extract_forces(str(out_file))

    number_of_na, number_of_cl, number_of_waters = get_number_of_na_cl_waters(str(structure_file))

    monomer_out_dir = Path(monomer_out_dir)
    e_na = extract_final_energy(str(monomer_out_dir / "na.out"))    if number_of_na     else 0.0
    e_cl = extract_final_energy(str(monomer_out_dir / "cl.out"))    if number_of_cl     else 0.0
    e_w  = extract_final_energy(str(monomer_out_dir / "water.out")) if number_of_waters else 0.0
    #print(energy_self, e_w)
    re = energy_self - number_of_na * e_na - number_of_cl * e_cl - number_of_waters * e_w

    # charges из precomputed chg
    chg_path = Path(chg_root) / scheme / f"{stem}.chg"
    if not chg_path.exists():
        raise RuntimeError(f"Missing precomputed charges: {chg_path}")

    coords, species, _ = read_xyz(str(structure_file))
    nat = len(coords)

    charges = parse_multiwfn_chg(chg_path, nat=nat)
    if len(charges) != nat:
        raise RuntimeError(f"#atoms mismatch: charges={len(charges)} vs xyz nat={nat}")

    structure_ase = Atoms(symbols=species, positions=coords)
    structure_ase.set_pbc(False)
    _, muDs = calculate_charges_dipole(structure_ase, species, coords, charges)
    
    #print(charges, muDs, structure_file)

    return float(re), np.array(forces), np.array(muDs), np.array(charges)

def load_dft_reference_dataset_with_scheme(
    *,
    coord_files,
    dataset_path: str | Path,
    dft_out_dir: str | Path,
    chg_root: str | Path,
    scheme: str,
    monomer_out_dir: str | Path,
):
    dataset_path = Path(dataset_path)
    res_ref, forces_ref, dipoles_ref, charges_ref = [], [], [], []

    for _, filename in coord_files:
        xyz_path = dataset_path / filename
        re, F, mu, q = calculate_re_f_md_dft_with_precomputed_chg(
            xyz_path,
            dft_out_dir=dft_out_dir,
            chg_root=chg_root,
            scheme=scheme,
            monomer_out_dir=monomer_out_dir,
        )
        res_ref.append(re)
        forces_ref.append(np.array(F))
        dipoles_ref.append(mu)
        charges_ref.append(q)

    res_ref = np.array(res_ref, dtype=float)
    N_atoms_ref = np.array([f.shape[0] for f in forces_ref], dtype=float)
    if np.any(N_atoms_ref == 0):
        raise ValueError("Encountered structure with zero atoms.")

    return res_ref, forces_ref, dipoles_ref, charges_ref

