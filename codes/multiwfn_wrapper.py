import re
import numpy as np
import os
import subprocess
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Tuple
import warnings

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_re_last_float = re.compile(rf".*?\s+({_FLOAT})\s*$")
_RE_TRSP_DEV = re.compile(
    r"Deviation of Tr\(S\*P\) to the total number of electrons:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
)
_RE_NTHREADS = re.compile(r"^(\s*nthreads\s*=\s*)\d+(\s*//.*)?$", re.MULTILINE)

def _ensure_atmrad(run_dir: Path, atmrad_dir: str | Path, mode: str = "symlink"):
    """
    Make sure run_dir/atmrad exists.
    mode: "symlink" (fast) or "copy" (most robust on weird FS)
    """
    atmrad_dir = Path(atmrad_dir)
    if not atmrad_dir.exists():
        raise RuntimeError(f"atmrad_dir is not found: {atmrad_dir}")

    target = run_dir / "atmrad"
    if target.exists():
        return

    if mode == "symlink":
        try:
            target.symlink_to(atmrad_dir, target_is_directory=True)
        except Exception:
            # fallback to copy if symlink not allowed
            shutil.copytree(atmrad_dir, target)
    elif mode == "copy":
        shutil.copytree(atmrad_dir, target)
    else:
        raise ValueError("mode must be 'symlink' or 'copy'")

def _multiwfn_bin_dir(multiwfn_exe: str | Path) -> Path:
    resolved = shutil.which(str(multiwfn_exe))
    exe_path = Path(resolved) if resolved is not None else Path(multiwfn_exe)
    return exe_path.expanduser().resolve().parent

def _set_multiwfn_thread_env(env: dict, multiwfn_exe: str | Path, omp_threads: int):
    threads = str(int(omp_threads))
    env["OMP_NUM_THREADS"] = threads
    env["OMP_THREAD_LIMIT"] = threads
    env["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "false")
    env["OMP_STACKSIZE"] = os.environ.get("OMP_STACKSIZE", "2G")
    env["KMP_STACKSIZE"] = os.environ.get("KMP_STACKSIZE", "2G")
    env["Multiwfnpath"] = str(_multiwfn_bin_dir(multiwfn_exe))
    return env

def _write_multiwfn_settings(run_dir: Path, multiwfn_exe: str | Path, omp_threads: int) -> Path:
    """Put a per-run settings.ini next to molden.input so Multiwfn uses the requested threads."""
    settings_src = _multiwfn_bin_dir(multiwfn_exe) / "settings.ini"
    if not settings_src.exists():
        raise RuntimeError(f"Multiwfn settings.ini is not found: {settings_src}")

    threads = int(omp_threads)
    settings_text = settings_src.read_text(encoding="utf-8", errors="ignore")
    if _RE_NTHREADS.search(settings_text):
        settings_text = _RE_NTHREADS.sub(rf"\g<1>{threads}\g<2>", settings_text, count=1)
    else:
        settings_text += f"\n  nthreads= {threads} // patched by ML-xTB workflow\n"

    settings_path = Path(run_dir) / "settings.ini"
    settings_path.write_text(settings_text, encoding="utf-8")
    return settings_path

def _check_multiwfn_output(stdout_text: str, *, trsp_tol: float = 1e-3, fatal_problem_wfn: bool = True):
    # 1) "problematic wavefunction" — обычно лучше считать фатальным
    if "Warning: The wavefunction loaded is problematic!" in stdout_text:
        msg_tail = stdout_text[-6000:]
        if fatal_problem_wfn:
            raise RuntimeError(
                "Multiwfn reports a problematic wavefunction / unsupported Molden file.\n"
                "See Multiwfn manual sections mentioned in the warning (CP2K: 2.9.2.1; Molden2AIM: 5.1).\n"
                f"--- last output ---\n{msg_tail}\n"
            )
        else:
            warnings.warn(
                "Multiwfn: problematic wavefunction warning detected. Results may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

    # 2) Deviation of Tr(S*P) ... 
    m = _RE_TRSP_DEV.search(stdout_text)
    if m:
        dev = float(m.group(1))
        if abs(dev) > trsp_tol:
            msg_tail = stdout_text[-6000:]
            raise RuntimeError(
                f"Multiwfn: |Deviation of Tr(S*P)| = {dev:.3e} exceeds tolerance {trsp_tol:.3e}.\n"
                f"--- last output ---\n{msg_tail}\n"
            )
        else:
            warnings.warn(
                f"Multiwfn: Deviation of Tr(S*P) = {dev:.3e} (<= {trsp_tol:.3e}).",
                RuntimeWarning,
                stacklevel=2,
            )

def count_atoms_in_molden(molden_path: str | Path) -> int:
    """Count atoms from [Atoms] section in a Molden file."""
    p = Path(molden_path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_atoms = False
    n = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[Atoms]"):
            in_atoms = True
            continue
        if in_atoms:
            if s.startswith("[") and not s.startswith("[Atoms]"):
                break
            # typical atom line: El idx Z x y z  (at least 6 columns)
            parts = s.split()
            if len(parts) >= 6:
                n += 1
    if n == 0:
        raise RuntimeError(f"Couldn't count [Atoms] in {p}")
    return n

def parse_multiwfn_chg(chg_path: str | Path, nat: Optional[int] = None) -> np.ndarray:
    """
    Parse Multiwfn .chg: take last column as charge for nat atoms.
    Robust to headers/comments: picks lines that end with a float.
    """
    p = Path(chg_path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    charges = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = _re_last_float.match(s)
        if not m:
            continue
        try:
            q = float(m.group(1))
        except Exception:
            continue
        charges.append(q)

    if nat is not None:
        if len(charges) < nat:
            raise RuntimeError(
                f"In {p} found only {len(charges)} float-lines, expected min nat={nat}. "
                f"Not correct file or Multiwfn did not write charges."
            )
        # обычно в chg ровно nat строк, но на всякий случай берём последние nat
        charges = charges[-nat:]

        if len(charges) != nat:
            raise RuntimeError(f"len(charges)={len(charges)} != nat={nat}")

    return np.array(charges, dtype=float)


'''
def _link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)          # hardlink (быстро и надёжно)
    except Exception:
        try:
            dst.symlink_to(src)    # если hardlink нельзя
        except Exception:
            shutil.copyfile(src, dst)  # fallback
'''

def _link_or_copy(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)

    # dst уже существует → если это тот же файл, ничего не делаем,
    # иначе удаляем и пересоздаём.
    if dst.exists() or dst.is_symlink():
        try:
            if os.path.samefile(src, dst):
                return
        except Exception:
            pass
        try:
            dst.unlink()
        except Exception:
            pass

    try:
        os.link(src, dst)          # hardlink
    except Exception:
        try:
            dst.symlink_to(src)    # symlink
        except Exception:
            shutil.copyfile(src, dst)

def run_multiwfn_charges(
    molden_path: str | Path,
    *,
    multiwfn_exe: str = "Multiwfn",
    timeout_s: int = 1800,
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    omp_threads: int = 1,
    out_chg_path: str | Path | None = None,
    debug: bool = False,
    atmrad_dir: str | Path | None = None,
    scheme: str = 'hirshfeldI',
) -> np.ndarray:
    """
    Run Multiwfn to compute Hirschfeld-I charges from a Molden file and return charges (nat,).

    - Copies molden to a run directory as 'molden.input'
    - Runs menu sequence: 7 -> 15/16/11 -> 1 -> y (save) -> 0 -> q
    - Reads 'molden.chg' and returns last column

    Parameters
    ----------
    molden_path : path to .molden
    multiwfn_exe : Multiwfn executable name/path
    timeout_s : kill if hangs
    workdir : directory to run in (if None -> TemporaryDirectory)
    keep_workdir : if using temp dir, keep it (writes path to stdout only if debug)
    omp_threads : set OMP_NUM_THREADS for Multiwfn
    out_chg_path : if set, copy resulting molden.chg there too
    debug : if True, prints Multiwfn output on failure
    scheme: choose partial charge scheme: 'hirshfeldI', 'CM5', 'ADCH'

    Returns
    -------
    charges : np.ndarray (nat,)
    """
    molden_path = Path(molden_path)
    nat = count_atoms_in_molden(molden_path)

    match scheme:
        case 'hirshfeldI':
            menu_item = '15'
        case 'CM5':
            menu_item = '16'
        case 'ADCH':
            menu_item = '11'
        case _:
            raise ValueError(f"Unknown scheme: {scheme}, choose: hirshfeldI, CM5 or ADCH")

    # --- choose run dir
    temp_ctx = None
    if workdir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"multiwfn_{scheme}_")
        run_dir = Path(temp_ctx.name)
    else:
        run_dir = Path(workdir)
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # standardize input name for Multiwfn
        in_name = run_dir / "molden.input"
        #shutil.copyfile(molden_path, in_name)
        _link_or_copy(molden_path, in_name)

        # remove old output if exists
        chg_file = run_dir / "molden.chg"
        if chg_file.exists():
            try:
                chg_file.unlink()
            except Exception:
                pass

        # Non-interactive menu script.
        menu = "\n".join([
            "7",   # Population analysis
            menu_item, # choose charge scheme
            "1",   # Start calculation
            "y",   # Save charges to molden.chg 
            "0",   # Return (из charges-меню)
            "q",   # Quit Multiwfn
            ""
        ])

        env = os.environ.copy()
        _set_multiwfn_thread_env(env, multiwfn_exe, omp_threads)
        _write_multiwfn_settings(run_dir, multiwfn_exe, omp_threads)

        # before subprocess.run(...)
        if atmrad_dir is not None:
            _ensure_atmrad(run_dir, atmrad_dir, mode="symlink")

        # Run Multiwfn
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

        stdout_text = proc.stdout.decode("utf-8", errors="ignore")
        _check_multiwfn_output(stdout_text, trsp_tol=1e-3, fatal_problem_wfn=True)

        # --- если файл уже появился и корректно читается, считаем успехом даже при returncode != 0
        chg_file = run_dir / "molden.chg"
        if chg_file.exists():
            try:
                charges = parse_multiwfn_chg(chg_file, nat=nat)
                # скопировать куда нужно
                if out_chg_path is not None:
                    out_chg_path = Path(out_chg_path)
                    out_chg_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(chg_file, out_chg_path)
                return charges
            except Exception:
                # файл есть, но не парсится — тогда ниже покажем вывод и упадём
                pass

        # --- если файла нет (или он битый) — тогда это реально ошибка
        if proc.returncode != 0:
            msg = stdout_text[-6000:]
            raise RuntimeError(
                f"Multiwfn exited with code {proc.returncode} for {molden_path}\n"
                f"--- last output ---\n{msg}\n"
            )

        # если returncode==0, но файла всё равно нет
        msg = stdout_text[-6000:]
        raise RuntimeError(
            f"Multiwfn exited normally, but molden.chg was not created for {molden_path}\n"
            f"--- last output ---\n{msg}\n"
        )

        charges = parse_multiwfn_chg(chg_file, nat=nat)

        # optionally save/copy .chg next to original
        if out_chg_path is not None:
            out_chg_path = Path(out_chg_path)
            out_chg_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(chg_file, out_chg_path)

        # cleanup / keep
        if debug and temp_ctx is not None and keep_workdir:
            print(f"[debug] kept Multiwfn dir: {run_dir}")

        return charges

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Multiwfn timed out (timeout={timeout_s}s) for {molden_path}. "
            f"Increase timeout_s"
        ) from e

    finally:
        # remove temp dir unless requested to keep
        if temp_ctx is not None and not keep_workdir:
            temp_ctx.cleanup()
