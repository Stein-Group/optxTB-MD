import os
import re
import math
import tomllib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from ase import Atoms
from tblite.interface import Result

from .gradients import get_total_charge, ensure_param_file, read_xyz, calculate_charges_dipole

from pathlib import Path
import json

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_re_header = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*.*$")
_re_energy_line = re.compile(rf"^\s*({_FLOAT})\s*$")
_re_3floats = re.compile(rf"^\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$")

def parse_tblite_tag_energy_gradient(tag_path: str | Path, nat: int):
    """
    Parses tblite tagged output written by: tblite run --grad <tag_path> ...
    Returns:
      energy (float, Eh),
      gradient (nat,3) (Eh/Bohr)
    """
    tag_path = Path(tag_path)
    lines = tag_path.read_text().splitlines()

    energy = None
    grad = None

    i = 0
    while i < len(lines):
        ln = lines[i].strip()

        # Example:
        # energy             :real:0:
        #  -4.45097E+01
        if ln.lower().startswith("energy"):
            # next non-empty numeric line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                raise RuntimeError(f"Energy block header found but no value in {tag_path}")
            m = _re_energy_line.match(lines[j])
            if not m:
                raise RuntimeError(f"Failed to parse energy value near line {j} in {tag_path}")
            energy = float(m.group(1))
            i = j + 1
            continue

        # Example:
        # gradient           :real:2:3,24
        #   x y z
        #   ...
        if ln.lower().startswith("gradient"):
            g = np.zeros((nat, 3), dtype=float)
            j = i + 1
            k = 0
            while j < len(lines) and k < nat:
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                m = _re_3floats.match(s)
                if not m:
                    # reached next block (e.g. virial) or something unexpected
                    break
                g[k, 0] = float(m.group(1))
                g[k, 1] = float(m.group(2))
                g[k, 2] = float(m.group(3))
                k += 1
                j += 1

            if k != nat:
                raise RuntimeError(
                    f"Gradient block parsed only {k}/{nat} атомов in {tag_path}. "
                    f"Проверь, что запуск был с --grad и файл не обрезан."
                )
            grad = g
            i = j
            continue

        i += 1

    if energy is None:
        raise RuntimeError(f"Energy not found in {tag_path}")
    if grad is None:
        raise RuntimeError(f"Gradient not found in {tag_path}")

    return energy, grad

def gradient_to_forces(grad: np.ndarray) -> np.ndarray:
    # forces = - dE/dR
    return -grad




# -----------------------------
# Reconstruct molden file from npz 
# -----------------------------
L_FROM_SHELL = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

def _molden_block_size(shell: str, spherical: bool = True) -> int:
    l = L_FROM_SHELL[shell]
    if l <= 1:
        return 2*l + 1  # s=1, p=3
    if spherical:
        return 2*l + 1  # d=5, f=7, g=9
    return (l + 1) * (l + 2) // 2  # cartesian sizes: 6D, 10F, 15G

def _perm_tblite_m_to_molden(l: int) -> np.ndarray:
    if l == 0:
        return np.array([0], dtype=int)
    if l == 1:
        # tblite m=-1,0,+1  -> molden (px,py,pz) convention we want
        return np.array([2, 0, 1], dtype=int)

    # want [0, +1, -1, +2, -2, ...]
    perm = [0 + l]  # m=0
    for k in range(1, l + 1):
        perm.append(+k + l)
        perm.append(-k + l)
    return np.array(perm, dtype=int)

def parse_gto_shells_in_order(molden_text: str, spherical: bool = True) -> List[Tuple[str, int]]:
    """
    Return list of shells in AO order: [(shell_letter, block_size), ...]
    based on [GTO] section (until [MO]).
    """
    if "[GTO]" not in molden_text:
        raise ValueError("No [GTO] section found")

    gto_start = molden_text.find("[GTO]")
    mo_start = molden_text.find("[MO]")
    gto = molden_text[gto_start: mo_start if mo_start != -1 else None].splitlines()

    shells: List[Tuple[str, int]] = []
    in_gto = False
    for line in gto:
        s = line.strip()
        if s.upper() == "[GTO]":
            in_gto = True
            continue
        if not in_gto or not s:
            continue

        head = s.split()
        if head and head[0].lower() in L_FROM_SHELL:
            sh = head[0].lower()
            shells.append((sh, _molden_block_size(sh, spherical=spherical)))

    if not shells:
        raise RuntimeError("Failed to parse shells from [GTO].")
    return shells

def reorder_C_tblite_to_molden_with_shells(C: np.ndarray, shells: List[Tuple[str, int]]) -> np.ndarray:
    """
    Reorder AO rows of C from tblite's m=-l..+l within each shell
    to Molden-expected order.
    """
    norb = C.shape[0]
    expected = sum(sz for _, sz in shells)
    if expected != norb:
        raise ValueError(f"AO count mismatch: template implies {expected}, but C has {norb}")

    C2 = C.copy()
    off = 0
    for sh, sz in shells:
        l = L_FROM_SHELL[sh]
        perm = _perm_tblite_m_to_molden(l)
        if len(perm) != sz:
            raise RuntimeError(f"Permutation size {len(perm)} != block size {sz} for shell {sh}")
        block = C2[off:off + sz, :]
        C2[off:off + sz, :] = block[perm, :]
        off += sz

    return C2

def patch_atoms_valence_charges(molden_text: str, valence_map: dict) -> str:
    """
    Replace 3rd column in [Atoms] with valence electrons (Z_eff) if you want Multiwfn net charge to be "valence-consistent".
    """
    lines = molden_text.splitlines()
    out = []
    in_atoms = False
    for ln in lines:
        if ln.strip().startswith("[Atoms]"):
            in_atoms = True
            out.append(ln)
            continue
        if in_atoms:
            if ln.strip().startswith("[") and not ln.strip().startswith("[Atoms]"):
                in_atoms = False
                out.append(ln)
                continue
            if ln.strip():
                parts = ln.split()
                # typical: El  idx  Z  x y z
                if len(parts) >= 6 and parts[0] in valence_map:
                    parts[2] = str(valence_map[parts[0]])
                    ln = " ".join(parts)
        out.append(ln)
    return "\n".join(out) + "\n"

def insert_spherical_tags_before_mo(molden_text: str, tags=("[5D]", "[7F]", "[9G]")) -> str:
    """
    Insert [5D]/[7F]/[9G] as separate lines immediately BEFORE [MO].
    Do nothing if any of these tags already appear anywhere.
    """
    if "[MO]" not in molden_text:
        raise ValueError("Template has no [MO] section")

    up = molden_text.upper()
    for t in tags:
        if t.upper() in up:
            return molden_text

    mo_pos = molden_text.find("[MO]")
    prefix = molden_text[:mo_pos].rstrip("\n") + "\n" + "\n"
    suffix = molden_text[mo_pos:]
    tag_block = "\n".join(tags) + "\n"
    return prefix + tag_block + suffix

def _replace_mo_section(template_text: str, mo_txt: str) -> str:
    mo_start = template_text.find("[MO]")
    if mo_start < 0:
        raise ValueError("Template has no [MO] section")
    return template_text[:mo_start] + "[MO]\n" + mo_txt

def _format_mo_block(energies, occs, coeffs, spin_label: str, sym_suffix: str) -> str:
    norb = energies.shape[0]
    lines = []
    for i in range(norb):
        sym = f"{i+1}{sym_suffix}"
        lines.append(f"Sym= {sym:>6s}")
        lines.append(f"Ene= {energies[i]: .15f}")
        lines.append(f"Spin= {spin_label}")
        lines.append(f"Occup= {occs[i]: .8f}")
        col = coeffs[:, i]
        for ao, c in enumerate(col, start=1):
            lines.append(f"{ao:>12d}  {c: .15E}")
    return "\n".join(lines) + "\n"

# -----------------------------
# Reconstruct molden file from npz with [GTO] section if shells, ngauss and slater parameters are invoked (basis-defining parameters)
# -----------------------------
_BASIS_AFFECTING_KEYS = frozenset({"shells", "slater", "ngauss"})

def needs_generated_gto(param_changes: Optional[dict]) -> bool:
    """
    Return True only if requested parameter changes touch basis-defining fields
    that require regenerating the [GTO] section instead of reusing xTB template [GTO].
    """
    if not param_changes:
        return False

    for block in param_changes.values():
        if not isinstance(block, dict):
            continue
        if any(k in _BASIS_AFFECTING_KEYS for k in block):
            return True
    return False


def extract_prefix_before_basis(template_text: str) -> str:
    """
    Keep everything before the first basis/MO-related section:
    [GTO], [5D], [7F], [9G], [MO]
    """
    tags = ["[GTO]", "[5D]", "[7F]", "[9G]", "[MO]"]
    positions = [template_text.find(tag) for tag in tags if template_text.find(tag) != -1]
    if not positions:
        return template_text
    cut = min(positions)
    return template_text[:cut]


def parse_atom_symbols_from_atoms_section(molden_text: str) -> List[str]:
    lines = molden_text.splitlines()
    in_atoms = False
    symbols: List[str] = []

    for ln in lines:
        s = ln.strip()
        if s.startswith("[Atoms]"):
            in_atoms = True
            continue
        if in_atoms:
            if s.startswith("[") and not s.startswith("[Atoms]"):
                break
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 6:
                symbols.append(parts[0])

    if not symbols:
        raise ValueError("Could not parse atom symbols from [Atoms] section")
    return symbols


def insert_spherical_tags_block(tags=("[5D]", "[7F]", "[9G]")) -> str:
    return "\n".join(tags) + "\n"


def load_element_params_from_toml(param_toml_path: str | Path):
    with open(param_toml_path, "rb") as f:
        params = tomllib.load(f)

    if "element" in params:
        return params["element"]

    raise KeyError("Could not find top-level 'element' table in TOML")


STO_NG = {
    1: {
        "1s": {"alpha": [0.2709498091], "coeff": [1.0]},
        "2s": {"alpha": [0.1012151084], "coeff": [1.0]},
        "3s": {"alpha": [0.05296881757], "coeff": [1.0]},
        "2p": {"alpha": [0.1759666885], "coeff": [1.0]},
        "3p": {"alpha": [0.09113614253], "coeff": [1.0]},
        "3d": {"alpha": [0.1302270363], "coeff": [1.0]},
    },
    2: {
        "1s": {"alpha": [0.8518186635, 0.1516232927], "coeff": [0.4301284983, 0.6789135305]},
        "2s": {"alpha": [0.1292278611, 0.04908584205], "coeff": [0.7470867124, 0.2855980556]},
        "3s": {"alpha": [0.6694095822, 0.05837135094], "coeff": [-0.1529645716, 1.05137011]},
        "2p": {"alpha": [0.4323908358, 0.1069439065], "coeff": [0.4522627513, 0.6713122642]},
        "3p": {"alpha": [0.1458620964, 0.05664210742], "coeff": [0.5349653144, 0.5299607212]},
        "3d": {"alpha": [0.2777427345, 0.08336507714], "coeff": [0.4666137923, 0.6644706516]},
    },
    3: {
        "1s": {"alpha": [2.227660584, 0.4057711562, 0.1098175104], "coeff": [0.1543289673, 0.5353281423, 0.4446345422]},
        "2s": {"alpha": [2.581578398, 0.1567622104, 0.06018332272], "coeff": [-0.05994474934, 0.5960385398, 0.4581786291]},
        "3s": {"alpha": [0.5641487709, 0.06924421391, 0.03269529097], "coeff": [-0.1782577972, 0.8612761663, 0.2261841969]},
        "2p": {"alpha": [0.9192379002, 0.2359194503, 0.08009805746], "coeff": [0.1623948553, 0.5661708862, 0.4223071752]},
        "3p": {"alpha": [2.692880368, 0.1489359592, 0.05739585040], "coeff": [-0.01061945788, 0.5218564264, 0.5450015143]},
        "3d": {"alpha": [0.5229112225, 0.1639595876, 0.06386630021], "coeff": [0.1686596060, 0.5847984817, 0.4056779523]},
    },
    4: {
        "1s": {"alpha": [5.216844534, 0.9546182760, 0.2652034102, 0.08801862774], "coeff": [0.0567524208, 0.2601413550, 0.5328461143, 0.2916254405]},
        "2s": {"alpha": [11.61525551, 2.000243111, 0.1607280687, 0.06125744532], "coeff": [-0.01198411747, -0.05472052539, 0.5805587176, 0.4770079976]},
        "3s": {"alpha": [1.513265591, 0.4262497508, 0.07643320863, 0.03760545063], "coeff": [-0.03295496352, -0.1724516959, 0.7518511194, 0.3589627317]},
        "2p": {"alpha": [1.798260992, 0.4662622228, 0.1643718620, 0.06543927065], "coeff": [0.05713170255, 0.2857455515, 0.5517873105, 0.2632314924]},
        "3p": {"alpha": [1.853180239, 0.1915075719, 0.08655487938, 0.04184253862], "coeff": [-0.01434249391, 0.2755177589, 0.5846750879, 0.2144986514]},
        "3d": {"alpha": [0.9185846715, 0.2920461109, 0.1187568890, 0.05286755896], "coeff": [0.05799057705, 0.3045581349, 0.5601358038, 0.2432423313]},
    },
    5: {
        "1s": {"alpha": [11.30563696, 2.071728178, 0.5786484833, 0.1975724573, 0.07445271746], "coeff": [0.02214055312, 0.1135411520, 0.3318161484, 0.4825700713, 0.1935721966]},
        "2s": {"alpha": [8.984956862, 1.673710636, 0.1944726668, 0.08806345634, 0.04249068522], "coeff": [-0.01596349096, -0.05685884883, 0.3698265599, 0.5480512593, 0.1472634893]},
        "3s": {"alpha": [4.275877914, 1.132409433, 0.4016256968, 0.07732370620, 0.03800708627], "coeff": [-0.003920358850, -0.04168430506, -0.1637440990, 0.7419373723, 0.3724364929]},
        "2p": {"alpha": [3.320386533, 0.8643257633, 0.3079819284, 0.1273309895, 0.05606243164], "coeff": [0.02079051117, 0.1235472099, 0.3667738886, 0.4834930290, 0.1653444074]},
        "3p": {"alpha": [6.466803859, 1.555914802, 0.1955925255, 0.08809647701, 0.04234835707], "coeff": [-0.002329023747, -0.01357395221, 0.2632185383, 0.5880427024, 0.2242794445]},
        "3d": {"alpha": [1.539033958, 0.4922090297, 0.2029756928, 0.09424112917, 0.04569058269], "coeff": [0.02020869128, 0.1321157923, 0.3911240346, 0.4779609701, 0.1463662294]},
    },
    6: {
        "1s": {"alpha": [23.10303149, 4.235915534, 1.185056519, 0.4070988982, 0.1580884151, 0.06510953954], "coeff": [0.009163596280, 0.04936149294, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841]},
        "2s": {"alpha": [27.68496241, 5.077140627, 1.426786050, 0.2040335729, 0.09260298399, 0.04416183978], "coeff": [-0.004151277819, -0.02067024148, -0.05150303337, 0.3346271174, 0.5621061301, 0.1712994697]},
        "3s": {"alpha": [3.273031938, 0.9200611311, 0.3593349765, 0.08636686991, 0.04797373812, 0.02724741144], "coeff": [-0.006775596947, -0.05639325779, -0.1587856086, 0.5534527651, 0.5015351020, 0.07223633674]},
        "2p": {"alpha": [5.868285913, 1.530329631, 0.5475665231, 0.2288932733, 0.1046655969, 0.04948220127], "coeff": [0.007924233646, 0.05144104825, 0.1898400060, 0.4049863191, 0.4012362861, 0.1051855189]},
        "3p": {"alpha": [5.077973607, 1.340786940, 0.2248434849, 0.1131741848, 0.06076408893, 0.03315424265], "coeff": [-0.003329929840, -0.01419488340, 0.1639395770, 0.4485358256, 0.3908813050, 0.07411456232]},
        "3d": {"alpha": [2.488296923, 0.7981487853, 0.3311327490, 0.1559114463, 0.07877734732, 0.04058484363], "coeff": [0.007283828112, 0.05386799363, 0.2072139149, 0.4266269092, 0.3843100204, 0.08902827546]},
    },
}

DFACTORIAL = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
TOP = 2.0 / math.pi

def parse_shell(shell: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d+)([spdfg])", shell.strip())
    if not m:
        raise ValueError(f"Cannot parse shell label: {shell!r}")
    n = int(m.group(1))
    l_char = m.group(2)
    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
    return n, l_map[l_char]


def fmt_xtb_gto_number(x: float) -> str:
    ax = abs(x)

    if x == 0.0:
        s = "0.0000000000000000"
        return f"{s:>21s}     "

    if ax < 0.1:
        s = np.format_float_scientific(
            x,
            unique=False,
            precision=16,
            exp_digits=3,
            trim="k",
        ).replace("e", "E")
        return f"{s:>26s}"
    else:
        s = np.format_float_positional(
            x,
            unique=False,
            precision=17,
            fractional=False,
            trim="k",
        )
        return f"{s:>21s}     "


def fmt_xtb_shell_header(l_char: str, ng: int) -> str:
    return f" {l_char:<1}{ng:12d}   1.0000000000000000     "


def tblite_sto_ng_shell(
    shell: str,
    zeta: float,
    ngauss: int,
    normalize: bool = True,
    xtb_molden_style: bool = True,
):
    n, l = parse_shell(shell)
    l_char = shell[-1]

    if ngauss not in STO_NG:
        raise ValueError(f"ngauss={ngauss} not supported here")

    if shell not in STO_NG[ngauss]:
        raise NotImplementedError(f"Shell {shell!r} is not included in STO_NG table.")

    base_alpha = STO_NG[ngauss][shell]["alpha"]
    base_coeff = STO_NG[ngauss][shell]["coeff"]

    alpha = [a * (zeta ** 2) for a in base_alpha]
    coeff = list(base_coeff)

    if normalize:
        normed = []
        for a, c in zip(alpha, coeff):
            fac = (TOP * a) ** 0.75
            fac *= (math.sqrt(4.0 * a) ** l) / math.sqrt(DFACTORIAL[l + 1])

            if xtb_molden_style and l > 0:
                fac *= math.sqrt(2 * l + 1)

            normed.append(c * fac)
        coeff = normed

    return l_char, alpha, coeff


def build_gto_section(
    atom_symbols: List[str],
    element_params,
    normalize: bool = True,
    xtb_molden_style: bool = True,
) -> str:
    lines = ["[GTO]"]

    for iat, sym in enumerate(atom_symbols, start=1):
        if sym not in element_params:
            raise KeyError(f"Element {sym!r} not found in parameter file")

        rec = element_params[sym]
        shells = rec["shells"]
        slater = rec["slater"]
        ngauss = rec["ngauss"]

        if not (len(shells) == len(slater) == len(ngauss)):
            raise ValueError(f"Inconsistent shell/slater/ngauss data for {sym}")

        lines.append(f"{iat:12d} 0")
        for sh, zeta, ng in zip(shells, slater, ngauss):
            l_char, alpha, coeff = tblite_sto_ng_shell(
                sh,
                float(zeta),
                int(ng),
                normalize=normalize,
                xtb_molden_style=xtb_molden_style,
            )

            lines.append(fmt_xtb_shell_header(l_char, int(ng)))
            for a, c in zip(alpha, coeff):
                lines.append(f"{fmt_xtb_gto_number(a)}{fmt_xtb_gto_number(c)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_molden_from_tblite_generated_gto(
    template: "MoldenTemplate",
    restart_path: Path,
    param_toml_path: str | Path,
    out_path: Path,
    add_spherical_tags: bool = True,
    valence_map: Optional[dict] = None,
    normalize_gto: bool = True,
    xtb_molden_style: bool = True,
) -> Path:
    """
    Full reconstruction path:
      - keep prefix from template only up to first [GTO]/[MO]
      - rebuild [GTO] from current TOML
      - rebuild [MO] from restart and reorder AO rows using the NEW [GTO]
    """
    prefix = extract_prefix_before_basis(template.text)
    if valence_map is not None:
        prefix = patch_atoms_valence_charges(prefix, valence_map)

    atom_symbols = parse_atom_symbols_from_atoms_section(prefix)
    element_params = load_element_params_from_toml(param_toml_path)

    gto_txt = build_gto_section(
        atom_symbols,
        element_params,
        normalize=normalize_gto,
        xtb_molden_style=xtb_molden_style,
    )

    shells = parse_gto_shells_in_order(gto_txt, spherical=True)
    has_dfg = any(sh in {"d", "f", "g"} for sh, _ in shells)

    res = Result()
    res.load(str(restart_path))

    eps = res.get("orbital-energies")
    occ = res.get("orbital-occupations")
    C = res.get("orbital-coefficients")

    if np.asarray(eps).ndim == 1:
        C2 = reorder_C_tblite_to_molden_with_shells(np.asarray(C), shells)
        mo_txt = _format_mo_block(np.asarray(eps), np.asarray(occ), C2, "Alpha", "a")
    else:
        Ca = reorder_C_tblite_to_molden_with_shells(np.asarray(C[0]), shells)
        Cb = reorder_C_tblite_to_molden_with_shells(np.asarray(C[1]), shells)
        mo_txt  = _format_mo_block(np.asarray(eps[0]), np.asarray(occ[0]), Ca, "Alpha", "a")
        mo_txt += _format_mo_block(np.asarray(eps[1]), np.asarray(occ[1]), Cb, "Beta",  "b")

    out_parts = [prefix.rstrip() + "\n", gto_txt, "\n"]
    if add_spherical_tags and has_dfg:
        out_parts.append(insert_spherical_tags_block())
    out_parts.append("[MO]\n")
    out_parts.append(mo_txt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_parts), encoding="utf-8")
    return out_path


def write_molden_from_tblite(
    template: "MoldenTemplate",
    restart_path: Path,
    out_path: Path,
    *,
    param_toml_path: str | Path | None = None,
    param_changes: Optional[dict] = None,
    add_spherical_tags: bool = True,
    valence_map: Optional[dict] = None,
    normalize_gto: bool = True,
    xtb_molden_style: bool = True,
) -> Path:
    """
    Dispatcher preserving the old behavior by default.

    Old path  : template [GTO] + rebuilt [MO]
    New path  : rebuilt [GTO] + rebuilt [MO]
    Triggered only when param_changes touch shells/slater/ngauss.
    """
    if needs_generated_gto(param_changes):
        if param_toml_path is None:
            raise ValueError("param_toml_path is required when regenerating [GTO]")
        return write_molden_from_tblite_generated_gto(
            template=template,
            restart_path=restart_path,
            param_toml_path=param_toml_path,
            out_path=out_path,
            add_spherical_tags=add_spherical_tags,
            valence_map=valence_map,
            normalize_gto=normalize_gto,
            xtb_molden_style=xtb_molden_style,
        )

    return write_molden_from_tblite_cached(
        template=template,
        restart_path=restart_path,
        out_path=out_path,
        add_spherical_tags=add_spherical_tags,
        valence_map=valence_map,
    )


@dataclass(frozen=True)
class MoldenTemplate:
    template_path: Path
    text: str
    shells_spherical: List[Tuple[str, int]]
    has_dfg: bool  # whether template contains d/f/g shells

def load_molden_template(template_path: Path, spherical: bool = True) -> MoldenTemplate:
    txt = template_path.read_text(encoding="utf-8", errors="ignore")
    if "[MO]" not in txt or "[GTO]" not in txt:
        raise RuntimeError(f"Bad template: {template_path}")

    shells = parse_gto_shells_in_order(txt, spherical=spherical)
    has_dfg = any(sh in {"d", "f", "g"} for sh, _ in shells)

    return MoldenTemplate(
        template_path=template_path,
        text=txt,
        shells_spherical=shells,
        has_dfg=has_dfg,
    )

class MoldenTemplateCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, int], MoldenTemplate] = {}

    def add(self, xyz_stem: str, charge: int, template_path: Path):
        self._cache[(xyz_stem, charge)] = load_molden_template(template_path, spherical=True)

    def get(self, xyz_stem: str, charge: int) -> MoldenTemplate:
        try:
            return self._cache[(xyz_stem, charge)]
        except KeyError:
            raise KeyError(f"No Molden template cached for ({xyz_stem}, charge={charge}).")

def write_molden_from_tblite_cached(
    template: MoldenTemplate,
    restart_path: Path,
    out_path: Path,
    add_spherical_tags: bool = True,
    valence_map: Optional[dict] = None,
) -> Path:
    """
    Build molden by taking template header/[GTO]/[Atoms], then replacing [MO] with tblite MOs
    AFTER reordering AO rows (tblite m=-l..l -> Molden order).
    """
    templ = template.text

    # IMPORTANT: only insert [5D]/[7F]/[9G] if d/f/g exist; avoids touching pure sp cases.
    if add_spherical_tags and template.has_dfg:
        templ = insert_spherical_tags_before_mo(templ, tags=("[5D]", "[7F]", "[9G]"))

    if valence_map is not None:
        templ = patch_atoms_valence_charges(templ, valence_map)

    res = Result()
    res.load(str(restart_path))

    eps = res.get("orbital-energies")
    occ = res.get("orbital-occupations")
    C   = res.get("orbital-coefficients")

    # reorder AO rows
    if eps.ndim == 1:
        C2 = reorder_C_tblite_to_molden_with_shells(np.asarray(C), template.shells_spherical)
        mo_txt = _format_mo_block(np.asarray(eps), np.asarray(occ), C2, "Alpha", "a")
    else:
        # spin-polarized: eps, occ: (2,norb); C: (2,norb,norb)
        Ca = reorder_C_tblite_to_molden_with_shells(np.asarray(C[0]), template.shells_spherical)
        Cb = reorder_C_tblite_to_molden_with_shells(np.asarray(C[1]), template.shells_spherical)
        mo_txt  = _format_mo_block(np.asarray(eps[0]), np.asarray(occ[0]), Ca, "Alpha", "a")
        mo_txt += _format_mo_block(np.asarray(eps[1]), np.asarray(occ[1]), Cb, "Beta",  "b")

    out_txt = _replace_mo_section(templ, mo_txt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_txt, encoding="utf-8")
    return out_path



# -----------------------------
# generate default molden template from xTB GFN2-xTB
# -----------------------------

def _safe_unlink(p: Path):
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def prepare_molden_templates_gfn2(coord_files: List[Tuple[int, str]],
                                 dataset_path: str,
                                 template_dir: Path,
                                 xtb_exe: str = "xtb",
                                 overwrite: bool = False) -> MoldenTemplateCache:
    """
    One-time: generate default GFN2 templates via xtb --gfn2 --molden for each xyz (and its charge).
    Stores templates as: {template_dir}/{stem}__q{charge}.molden
    Returns: MoldenTemplateCache preloaded in memory.
    """
    dataset_path = str(dataset_path)
    template_dir.mkdir(parents=True, exist_ok=True)

    cache = MoldenTemplateCache()

    # Thread limiting, same idea as you did for DFTB+
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    for _, filename in coord_files:
        xyz_path = Path(dataset_path) / filename
        stem = xyz_path.stem

        # your function
        charge = get_total_charge(str(xyz_path))

        templ_path = template_dir / f"{stem}__q{charge:+d}.molden"
        if templ_path.exists() and not overwrite:
            cache.add(stem, charge, templ_path)
            continue

        run_dir = template_dir / "_xtb_runs" / f"{stem}__q{charge:+d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # xtb writes molden.input in cwd
        _safe_unlink(run_dir / "molden.input")
        _safe_unlink(run_dir / "xtb.out")

        cmd = [xtb_exe, str(xyz_path), "--gfn2", "--chrg", str(charge), "--molden"]

        with (run_dir / "xtb.out").open("w") as out:
            subprocess.run(cmd, check=True, cwd=str(run_dir), stdout=out, stderr=subprocess.STDOUT, env=env)

        produced = run_dir / "molden.input"
        if not produced.exists():
            raise RuntimeError(f"xtb did not produce molden.input for {xyz_path} in {run_dir}")

        shutil.copyfile(produced, templ_path)
        cache.add(stem, charge, templ_path)

    return cache


# -----------------------------
# run tblite per structure 
# -----------------------------


def run_tblite_cli(
    xyz_path: str | Path,
    run_dir: str | Path,
    param_toml_path: str | Path,
    charge: int = 0,
    iterations: int = 2000,
    restart_path: str | Path | None = None,
    restart_policy: str = "recreate",  # "keep" | "recreate" | "discard"
    omp_tblite = 1,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = Path(xyz_path)
    param_toml_path = Path(param_toml_path)

    tag_path = run_dir / "tblite.txt"
    json_path = run_dir / "tblite.json"
    out_path = run_dir / "tblite.out"

    if restart_path is None:
        restart_path = run_dir / "tblite-restart.npz"
    else:
        restart_path = Path(restart_path)

    # clean previous outputs
    for p in (tag_path, json_path, out_path):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    if restart_policy == "recreate" and restart_path.exists():
        try:
            restart_path.unlink()
        except Exception:
            pass

    # avoid oversubscription
    env = os.environ.copy()
    #env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", str(omp_tblite))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    cmd = [
        "tblite", "run",
        "--param", str(param_toml_path),
        "-c", str(charge),
        "--iterations", str(iterations),
        "--grad", str(tag_path),
        "--acc", str(0.0001),
        "--json", str(json_path),
        "--restart", str(restart_path),
        str(xyz_path),
    ]

    with out_path.open("w") as fout:
        subprocess.run(cmd, cwd=str(run_dir), check=True, stdout=fout, stderr=subprocess.STDOUT, env=env)

    if restart_policy == "discard" and restart_path.exists():
        try:
            restart_path.unlink()
        except Exception:
            pass

    return tag_path, json_path, restart_path, out_path



# -----------------------------
# Molden reconstruction
# -----------------------------




