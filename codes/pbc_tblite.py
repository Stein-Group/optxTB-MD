from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
ENERGY_STDOUT_PATTERNS = [
    re.compile(rf"total energy\s+({FLOAT_RE})", re.IGNORECASE),
    re.compile(rf"energy\s*[:=]\s*({FLOAT_RE})", re.IGNORECASE),
]
SPIN_BY_SYMBOL = {"H": 1, "O": 2, "Na": 1, "Cl": 1}


@dataclass
class TblitePBCResult:
    energy_ha: float
    forces_ha_bohr: np.ndarray | None
    virial_ha: np.ndarray | None
    tag_path: Path
    json_path: Path
    out_path: Path


def parameter_file_hash(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tblite_env(omp_threads: int = 1) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("OMP_PROC_BIND", "false")
    return env


def _read_numeric_block(lines: list[str], start: int, n_rows: int) -> tuple[np.ndarray, int]:
    rows: list[list[float]] = []
    j = start
    while j < len(lines) and len(rows) < n_rows:
        stripped = lines[j].strip()
        if not stripped:
            j += 1
            continue
        parts = stripped.split()
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            raise RuntimeError(f"Expected numeric row near tagged output line {j + 1}: {lines[j]!r}") from exc
        j += 1
    if len(rows) != n_rows:
        raise RuntimeError(f"Parsed only {len(rows)}/{n_rows} rows from tagged output")
    return np.asarray(rows, dtype=float), j


def parse_tblite_tag_energy_gradient_virial(tag_path: str | Path, nat: int) -> tuple[float, np.ndarray, np.ndarray]:
    tag_path = Path(tag_path)
    lines = tag_path.read_text().splitlines()
    energy = None
    gradient = None
    virial = None

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        key = stripped.split()[0].lower()

        if key == "energy":
            block, i = _read_numeric_block(lines, i + 1, 1)
            energy = float(block.reshape(-1)[0])
            continue
        if key == "gradient":
            gradient, i = _read_numeric_block(lines, i + 1, nat)
            if gradient.shape != (nat, 3):
                raise RuntimeError(f"{tag_path}: gradient shape {gradient.shape} != {(nat, 3)}")
            continue
        if key == "virial":
            virial, i = _read_numeric_block(lines, i + 1, 3)
            if virial.shape != (3, 3):
                raise RuntimeError(f"{tag_path}: virial shape {virial.shape} != {(3, 3)}")
            continue
        i += 1

    if energy is None:
        raise RuntimeError(f"Energy block not found in {tag_path}")
    if gradient is None:
        raise RuntimeError(f"Gradient block not found in {tag_path}")
    if virial is None:
        raise RuntimeError(f"Virial block not found in {tag_path}")
    return energy, gradient, virial


def parse_tblite_tag_energy_gradient(tag_path: str | Path, nat: int) -> tuple[float, np.ndarray]:
    tag_path = Path(tag_path)
    lines = tag_path.read_text().splitlines()
    energy = None
    gradient = None

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        key = stripped.split()[0].lower()

        if key == "energy":
            block, i = _read_numeric_block(lines, i + 1, 1)
            energy = float(block.reshape(-1)[0])
            continue
        if key == "gradient":
            gradient, i = _read_numeric_block(lines, i + 1, nat)
            if gradient.shape != (nat, 3):
                raise RuntimeError(f"{tag_path}: gradient shape {gradient.shape} != {(nat, 3)}")
            continue
        i += 1

    if energy is None:
        raise RuntimeError(f"Energy block not found in {tag_path}")
    if gradient is None:
        raise RuntimeError(f"Gradient block not found in {tag_path}")
    return energy, gradient


def run_tblite_molecule(
    *,
    xyz_path: str | Path,
    nat: int,
    param_toml_path: str | Path,
    run_dir: str | Path,
    charge: int = 0,
    spin: int | None = None,
    iterations: int = 2000,
    acc: float = 0.0001,
    etemp: float | None = None,
    compute_forces: bool = True,
    omp_threads: int = 1,
) -> tuple[float, np.ndarray | None]:
    """Single-point energy (+ optional forces) for a non-periodic xyz file (no virial).

    `etemp` raises the electronic temperature used for SCF occupation
    smearing (tblite default: 300 K); this helps SCF convergence for weakly
    interacting/near-dissociated fragments where orbitals near the Fermi
    level are close to degenerate (e.g. long-range scan points). Set
    `compute_forces=False` to skip the (more expensive) gradient evaluation
    when only the energy is needed.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = Path(xyz_path)
    param_toml_path = Path(param_toml_path)

    tag_path = run_dir / "tblite.txt"
    json_path = run_dir / "tblite.json"
    out_path = run_dir / "molecule_gfn2_custom.out"
    for path in (tag_path, json_path, out_path):
        if path.exists():
            path.unlink()

    cmd = [
        "tblite",
        "run",
        "--param",
        str(param_toml_path),
        "--charge",
        str(charge),
        "--iterations",
        str(iterations),
        "--acc",
        str(acc),
        "--json",
        str(json_path),
    ]
    if compute_forces:
        cmd += ["--grad", str(tag_path)]
    if spin is not None:
        cmd += ["--spin", str(spin), "--spin-polarized"]
    if etemp is not None:
        cmd += ["--etemp", str(etemp)]
    cmd.append(str(xyz_path))

    with out_path.open("w") as stdout:
        subprocess.run(
            cmd,
            cwd=str(run_dir),
            check=True,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=_tblite_env(omp_threads),
        )

    if not compute_forces:
        energy = _parse_energy_from_json(json_path)
        if energy is None:
            energy = _parse_energy_from_stdout(out_path)
        return float(energy), None

    energy, gradient = parse_tblite_tag_energy_gradient(tag_path, nat=nat)
    forces = -np.asarray(gradient, dtype=float)
    return float(energy), forces


def run_tblite_pbc(
    *,
    poscar_path: str | Path,
    nat: int,
    param_toml_path: str | Path,
    run_dir: str | Path,
    charge: int = 0,
    iterations: int = 2000,
    acc: float = 0.0001,
    omp_threads: int = 1,
    etemp: float | None = None,
    compute_forces: bool = True,
) -> TblitePBCResult:
    """`etemp` raises the electronic temperature used for SCF occupation
    smearing (tblite default: 300 K) - helps SCF convergence for
    near-degenerate/oscillating cases (see run_tblite_molecule).

    Set `compute_forces=False` to skip the (more expensive) gradient/virial
    evaluation when only the energy is needed (e.g. fitting composition
    offsets on a train split). `forces_ha_bohr`/`virial_ha` are None then.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    poscar_path = Path(poscar_path)
    param_toml_path = Path(param_toml_path)

    tag_path = run_dir / "tblite.txt"
    json_path = run_dir / "tblite.json"
    out_path = run_dir / "pbc_gfn2_custom.out"
    for path in (tag_path, json_path, out_path):
        if path.exists():
            path.unlink()

    cmd = [
        "tblite",
        "run",
        "--param",
        str(param_toml_path),
        "--charge",
        str(charge),
        "--iterations",
        str(iterations),
        "--acc",
        str(acc),
        "--json",
        str(json_path),
    ]
    if compute_forces:
        cmd += ["--grad", str(tag_path)]
    if etemp is not None:
        cmd += ["--etemp", str(etemp)]
    cmd.append(str(poscar_path))

    with out_path.open("w") as stdout:
        subprocess.run(
            cmd,
            cwd=str(run_dir),
            check=True,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=_tblite_env(omp_threads),
        )

    if not compute_forces:
        energy = _parse_energy_from_json(json_path)
        if energy is None:
            energy = _parse_energy_from_stdout(out_path)
        return TblitePBCResult(
            energy_ha=float(energy),
            forces_ha_bohr=None,
            virial_ha=None,
            tag_path=tag_path,
            json_path=json_path,
            out_path=out_path,
        )

    energy, gradient, virial = parse_tblite_tag_energy_gradient_virial(tag_path, nat=nat)
    return TblitePBCResult(
        energy_ha=float(energy),
        forces_ha_bohr=-np.asarray(gradient, dtype=float),
        virial_ha=np.asarray(virial, dtype=float),
        tag_path=tag_path,
        json_path=json_path,
        out_path=out_path,
    )


def _write_atom_xyz(path: Path, symbol: str) -> None:
    path.write_text(f"1\n{symbol} neutral atom\n{symbol} 0.0 0.0 0.0\n", encoding="utf-8")


def _parse_energy_from_json(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    if isinstance(data, dict) and "energy" in data:
        return float(data["energy"])
    return None


def _parse_energy_from_stdout(out_path: Path) -> float:
    text = out_path.read_text(errors="ignore")
    for pattern in ENERGY_STDOUT_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return float(matches[-1])
    raise RuntimeError(f"Could not parse tblite atom energy from {out_path}")


def run_tblite_isolated_atom(
    *,
    symbol: str,
    param_toml_path: str | Path,
    run_dir: str | Path,
    spin: int | None = None,
    charge: int = 0,
    omp_threads: int = 1,
) -> dict[str, str | float | int]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    param_toml_path = Path(param_toml_path)
    spin = SPIN_BY_SYMBOL[symbol] if spin is None else int(spin)

    xyz_path = run_dir / f"{symbol}.xyz"
    json_path = run_dir / "tblite.json"
    out_path = run_dir / "gfn2_custom_spinpol.out"
    _write_atom_xyz(xyz_path, symbol)
    for path in (json_path, out_path):
        if path.exists():
            path.unlink()

    cmd = [
        "tblite",
        "run",
        "--param",
        str(param_toml_path),
        "--charge",
        str(charge),
        "--spin",
        str(spin),
        "--acc",
        str(0.0001),
        "--spin-polarized",
        "--json",
        str(json_path),
        str(xyz_path),
    ]
    with out_path.open("w") as stdout:
        subprocess.run(
            cmd,
            cwd=str(run_dir),
            check=True,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=_tblite_env(omp_threads),
        )

    energy = _parse_energy_from_json(json_path)
    if energy is None:
        energy = _parse_energy_from_stdout(out_path)
    return {
        "symbol": symbol,
        "energy_ha": float(energy),
        "charge": int(charge),
        "spin": int(spin),
        "out_path": str(out_path),
        "json_path": str(json_path),
    }


def get_or_run_isolated_atom_energies(
    *,
    symbols: Iterable[str],
    param_toml_path: str | Path,
    cache_root: str | Path,
    omp_threads: int = 1,
) -> dict[str, dict[str, str | float | int]]:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    param_hash = parameter_file_hash(param_toml_path)
    cache_path = cache_root / "method_atom_energies.json"
    symbols = sorted(set(symbols))

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("param_sha256") == param_hash and sorted(cached.get("energies", {})) == symbols:
            return cached["energies"]

    energies: dict[str, dict[str, str | float | int]] = {}
    for symbol in symbols:
        energies[symbol] = run_tblite_isolated_atom(
            symbol=symbol,
            param_toml_path=param_toml_path,
            run_dir=cache_root / f"atom_{symbol}",
            omp_threads=omp_threads,
        )

    cache_path.write_text(
        json.dumps({"param_sha256": param_hash, "energies": energies}, indent=2),
        encoding="utf-8",
    )
    return energies
