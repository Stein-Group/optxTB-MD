from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase.io import read, write


GPA_A3_TO_HA = 1.0e9 * 1.0e-30 / 4.3597447222071e-18
REFERENCE_VIRIAL_SIGN = -1.0
CP2K_FINAL_ENERGY_RE = re.compile(
    r"ENERGY\|\s+Total\s+FORCE_EVAL.*?\[hartree\]\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
)
CP2K_TOTAL_ENERGY_RE = re.compile(
    r"Total energy:\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
)


@dataclass
class PBCStructureRecord:
    frame_id: str
    frame_index: int
    extxyz_path: Path
    atoms: object
    n_atoms: int
    counts: dict[str, int]
    metadata: dict
    ref_energy_ha: float
    ref_forces_ha_bohr: np.ndarray
    ref_stress_gpa: np.ndarray
    ref_virial_ha: np.ndarray
    poscar_path: Path | None = None


def _as_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _require_unit(info: dict, key: str, expected: str, path: Path) -> None:
    actual = str(_as_scalar(info.get(key, ""))).strip().lower()
    if actual != expected.lower():
        raise ValueError(f"{path.name}: expected {key}={expected!r}, got {actual!r}")


def _parse_frame_index(path: Path, atoms) -> int:
    value = atoms.info.get("frame_index", atoms.info.get("global_index"))
    if value is not None:
        return int(_as_scalar(value))
    match = re.search(r"frame_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not infer frame index from {path}")
    return int(match.group(1))


def stress_gpa_to_virial_ha(stress_gpa: np.ndarray, volume_a3: float) -> np.ndarray:
    """Convert CP2K stress to virial in Ha.

    Convention used here is Xi = -V * stress.  If later comparison shows the
    CP2K/tblite convention is opposite for this dataset, change
    REFERENCE_VIRIAL_SIGN in this module only.
    """
    return REFERENCE_VIRIAL_SIGN * np.asarray(stress_gpa, dtype=float) * volume_a3 * GPA_A3_TO_HA


def classify_system(atoms) -> str:
    n_na = int(_as_scalar(atoms.info.get("n_Na", 0)))
    n_cl = int(_as_scalar(atoms.info.get("n_Cl", 0)))
    if n_na == 0 and n_cl == 0:
        return "water"
    return "nacl"


def load_pbc_dataset(
    dataset_dir: str | Path,
    *,
    system_filter: str = "water",
    max_structures: int | None = None,
) -> list[PBCStructureRecord]:
    dataset_dir = Path(dataset_dir)
    paths = sorted(dataset_dir.glob("frame_*.extxyz"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.extxyz files found in {dataset_dir}")

    records: list[PBCStructureRecord] = []
    for path in paths:
        atoms = read(path)
        system_type = classify_system(atoms)
        if system_filter != "all" and system_type != system_filter:
            continue

        info = atoms.info
        _require_unit(info, "cp2k_energy_units", "hartree", path)
        _require_unit(info, "cp2k_forces_units", "hartree/bohr", path)
        _require_unit(info, "positions_units", "angstrom", path)
        _require_unit(info, "lattice_units", "angstrom", path)
        _require_unit(info, "cp2k_stress_units", "GPa", path)

        stress_order = str(info.get("cp2k_stress_order", "")).strip()
        if stress_order != "xx xy xz yx yy yz zx zy zz":
            raise ValueError(f"{path.name}: unsupported stress order {stress_order!r}")

        forces = np.asarray(atoms.arrays.get("cp2k_forces"), dtype=float)
        if forces.shape != (len(atoms), 3):
            raise ValueError(f"{path.name}: cp2k_forces shape {forces.shape} != {(len(atoms), 3)}")

        stress = np.asarray(info["cp2k_stress_tensor_GPa"], dtype=float).reshape(3, 3)
        virial = stress_gpa_to_virial_ha(stress, float(atoms.get_volume()))
        symbols = atoms.get_chemical_symbols()
        counts = dict(Counter(symbols))
        frame_index = _parse_frame_index(path, atoms)

        records.append(
            PBCStructureRecord(
                frame_id=f"frame_{frame_index:06d}",
                frame_index=frame_index,
                extxyz_path=path,
                atoms=atoms,
                n_atoms=len(atoms),
                counts=counts,
                metadata={k: _as_scalar(v) for k, v in info.items()},
                ref_energy_ha=float(_as_scalar(info["cp2k_energy_hartree"])),
                ref_forces_ha_bohr=forces,
                ref_stress_gpa=stress,
                ref_virial_ha=virial,
            )
        )

        if max_structures is not None and len(records) >= max_structures:
            break

    return records


def ensure_poscar_cache(
    records: Iterable[PBCStructureRecord],
    cache_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[PBCStructureRecord]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_records: list[PBCStructureRecord] = []

    for record in records:
        frame_dir = cache_dir / record.frame_id
        poscar_path = frame_dir / "POSCAR"
        frame_dir.mkdir(parents=True, exist_ok=True)

        if overwrite or not poscar_path.exists() or poscar_path.stat().st_size == 0:
            tmp_path = frame_dir / f".POSCAR.tmp.{os.getpid()}"
            write(tmp_path, record.atoms, format="vasp", vasp5=True, direct=False)
            os.replace(tmp_path, poscar_path)

        record.poscar_path = poscar_path
        out_records.append(record)

    return out_records


def parse_cp2k_isolated_atom_energy(atom_dir: str | Path, *, allow_total_fallback: bool = False) -> tuple[float, str]:
    atom_dir = Path(atom_dir)
    log_path = atom_dir / "input.inp.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing isolated atom CP2K log: {log_path}")

    text = log_path.read_text(errors="ignore")
    final_matches = CP2K_FINAL_ENERGY_RE.findall(text)
    if final_matches:
        return float(final_matches[-1]), f"{log_path}: ENERGY| Total FORCE_EVAL"

    if allow_total_fallback:
        total_matches = CP2K_TOTAL_ENERGY_RE.findall(text)
        if total_matches:
            return float(total_matches[-1]), f"{log_path}: fallback last Total energy"

    raise RuntimeError(f"No final CP2K ENERGY| Total FORCE_EVAL line found in {log_path}")


def load_reference_atom_energies(
    isolated_atoms_dir: str | Path,
    symbols: Iterable[str],
    *,
    allow_total_fallback: bool = False,
) -> dict[str, dict[str, str | float]]:
    isolated_atoms_dir = Path(isolated_atoms_dir)
    result: dict[str, dict[str, str | float]] = {}
    for symbol in sorted(set(symbols)):
        energy, source = parse_cp2k_isolated_atom_energy(
            isolated_atoms_dir / f"atom_{symbol}",
            allow_total_fallback=allow_total_fallback,
        )
        result[symbol] = {"energy_ha": float(energy), "source": source}
    return result


def composition_matrix(records: Iterable[PBCStructureRecord], elements=("H", "O", "Na", "Cl")) -> np.ndarray:
    records = list(records)
    return np.asarray(
        [[float(record.counts.get(element, 0)) for element in elements] for record in records],
        dtype=float,
    )


def composition_residuals(
    total_energies: Iterable[float],
    records: Iterable[PBCStructureRecord],
    offsets: dict[str, float] | np.ndarray,
    elements=("H", "O", "Na", "Cl"),
) -> np.ndarray:
    total_energies = np.asarray(list(total_energies), dtype=float)
    matrix = composition_matrix(records, elements=elements)
    if isinstance(offsets, dict):
        offset_vector = np.asarray([float(offsets[element]) for element in elements], dtype=float)
    else:
        offset_vector = np.asarray(offsets, dtype=float)
    return total_energies - matrix @ offset_vector


def fit_composition_offsets(
    total_energies: Iterable[float],
    records: Iterable[PBCStructureRecord],
    elements=("H", "O", "Na", "Cl"),
    rcond=None,
) -> dict:
    records = list(records)
    total_energies = np.asarray(list(total_energies), dtype=float)
    matrix = composition_matrix(records, elements=elements)
    offsets, _residual_sum, rank, singular_values = np.linalg.lstsq(matrix, total_energies, rcond=rcond)
    residuals = total_energies - matrix @ offsets
    return {
        "elements": list(elements),
        "offsets": {element: float(value) for element, value in zip(elements, offsets)},
        "offset_vector": offsets,
        "residuals": residuals,
        "rank": int(rank),
        "singular_values": [float(value) for value in singular_values],
    }


def atomization_energy(total_energy_ha: float, counts: dict[str, int], atom_energies_ha: dict[str, float]) -> float:
    missing = sorted(symbol for symbol in counts if symbol not in atom_energies_ha)
    if missing:
        raise KeyError(f"Missing isolated atom energies for {missing}")
    return float(total_energy_ha) - sum(int(n) * float(atom_energies_ha[symbol]) for symbol, n in counts.items())
