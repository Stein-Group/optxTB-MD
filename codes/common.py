"""
Helper functions shared by the FIM and NSGA driver scripts in optimization/.

Each function is identical to the copy that previously lived in the individual
driver scripts; they are collected here so that every driver uses the same
implementation.
"""

import csv
import os
import platform
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml
from scipy.stats import qmc
from sklearn.metrics import r2_score, root_mean_squared_error


def _env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def init_tblite_env(omp_threads=1, blas_threads=1, amd_mkl=False):
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    os.environ.setdefault("OMP_PROC_BIND", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("BLIS_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(blas_threads))
    if platform.system() == "Darwin":
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(blas_threads))
    if amd_mkl:
        os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")


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

    names = []
    bounds = []
    vector = []
    structure = {}
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


def load_fixed_params_from_toml(path, fixed_element_params, fixed_kpair_keys):
    params = toml.load(path)
    fixed = {"kpair": {}}
    kpair = params["hamiltonian"]["xtb"]["kpair"]
    for key in fixed_kpair_keys:
        fixed["kpair"][key] = kpair[key]
    for elem, keys in fixed_element_params.items():
        fixed[elem] = {}
        elem_params = params["element"][elem]
        for key in keys:
            fixed[elem][key] = elem_params[key]
    return fixed


def merge_param_changes(fixed_changes, optimized_changes):
    merged = {}
    for elem, params in fixed_changes.items():
        merged[elem] = {key: list(val) if isinstance(val, list) else val for key, val in params.items()}
    for elem, params in optimized_changes.items():
        merged.setdefault(elem, {})
        for key, val in params.items():
            merged[elem][key] = list(val) if isinstance(val, list) else val
    return merged


def is_na_tier(file_number):
    """T2_Na (Na+ in water) or T3 (NaCl ion pair in water)."""
    return (82 <= file_number <= 180) or (290 <= file_number <= 307)


def parse_isomer_and_distortion(filename_base):
    base = filename_base
    if 'mode' in base:
        prefix, mode_part = base.split('mode', 1)
        isomer_id = prefix.rstrip('_')
        m = re.match(r'(\d+)_?(minus|plus)', mode_part)
        if m:
            mode_id = int(m.group(1))
            sign = m.group(2)
        else:
            mode_id = None
            sign = None
    else:
        isomer_id = base
        mode_id = None
        sign = 'eq'  # original geometry, undistorted
    return isomer_id.lower(), mode_id, sign


def split_nacl_dataset(records, train_frac, seed):
    """Stratified split by (IP_label, n_atoms) so each stratum is split proportionally."""
    strata = defaultdict(list)
    for record in records:
        label = record.metadata.get("IP_label", "unknown")
        strata[(label, record.n_atoms)].append(record)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for key, group in sorted(strata.items()):
        group = list(group)
        rng.shuffle(group)
        n_tr = max(1, round(train_frac * len(group)))
        train.extend(group[:n_tr])
        test.extend(group[n_tr:])
    return train, test


def _vector_cache_key(vector, decimals=14):
    return tuple(np.round(np.asarray(vector, dtype=float), decimals=decimals))


def generate_sobol_initial_population(bounds, pop_size, default_vector, seed=42, scramble=True):
    lower_bounds = np.array([b[0] for b in bounds], dtype=float)
    upper_bounds = np.array([b[1] for b in bounds], dtype=float)
    dim = len(bounds)
    X_init = np.empty((pop_size, dim), dtype=float)
    X_init[0] = np.asarray(default_vector, dtype=float)

    n_sobol = pop_size - 1
    if n_sobol > 0:
        m = int(np.log2(n_sobol))
        if 2 ** m != n_sobol:
            raise ValueError(f"pop_size - 1 must be a power of two for Sobol random_base2; got {n_sobol}.")
        sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        unit_points = sampler.random_base2(m=m)
        sobol_points = qmc.scale(unit_points, lower_bounds, upper_bounds)
        print("Deterministic Sobol sample shape:", sobol_points.shape)
        X_init[1:] = sobol_points
    return X_init


def nondominated_indices(F):
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return np.array([], dtype=int)
    keep = np.ones(F.shape[0], dtype=bool)
    for i in range(F.shape[0]):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return np.where(keep)[0]


def compute_fim_from_jacobian(J, slices=None):
    out = {"total": J.T @ J}
    if slices is not None:
        for key, sl in slices.items():
            Jk = J[sl, :]
            out[key] = Jk.T @ Jk
    return out


def eigh_sorted(F, descending=True):
    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)
    if descending:
        order = order[::-1]
    return evals[order], evecs[:, order]


def fim_condition_info(evals, tol=1e-12):
    evals = np.asarray(evals, dtype=float)
    lam_max = float(np.max(evals))
    lam_min = float(np.min(evals))
    nonzero = evals[evals > tol]
    cond = np.inf if nonzero.size == 0 else lam_max / float(np.min(nonzero))
    return {"lambda_max": lam_max, "lambda_min": lam_min, "condition_number_nonzero": float(cond), "n_below_tol": int(np.sum(evals <= tol))}


def summarize_eigenvectors(evals, evecs, names, n_show=5, top_loadings=6, which="stiff"):
    p = len(evals)
    if which == "stiff":
        idx_dirs = range(min(n_show, p))
    elif which == "sloppy":
        idx_dirs = range(max(0, p - n_show), p)
    else:
        raise ValueError("which must be 'stiff' or 'sloppy'")

    rows = []
    print(f"\n===== {which.upper()} directions =====")
    for k in idx_dirs:
        lam = float(evals[k])
        vec = evecs[:, k]
        idx = np.argsort(np.abs(vec))[::-1][:top_loadings]
        print(f"\nDirection {k}: eigenvalue = {lam:.6e}")
        for rank, ii in enumerate(idx, start=1):
            coeff = float(vec[ii])
            print(f"   {names[ii]:20s}  coeff = {coeff: .4f}")
            rows.append({
                "which": which,
                "mode": int(k),
                "eigenvalue": lam,
                "rank": rank,
                "parameter": names[ii],
                "coefficient": coeff,
                "participation": coeff * coeff,
            })
    return rows


def write_csv_rows(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_spectrum(evals, label, out_path):
    evals = np.asarray(evals, dtype=float)
    evals_pos = np.clip(evals, 1e-30, None)
    rel = evals_pos / evals_pos[0]
    total = np.sum(evals_pos)
    cum = np.cumsum(evals_pos) / total if total > 0 else np.full_like(evals_pos, np.nan)
    modes = np.arange(len(evals))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(modes, rel, marker="o"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Mode"); axes[0].set_ylabel("lambda_k / lambda_max")
    axes[0].set_title(f"Relative FIM spectrum ({label})"); axes[0].grid(True)
    axes[1].plot(modes, cum, marker="o")
    axes[1].set_xlabel("Mode"); axes[1].set_ylabel("Cumulative trace fraction")
    axes[1].set_title(f"Cumulative information ({label})"); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def parity_plot(x, y, xlabel, ylabel, title, out_path):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(x, y, s=16, alpha=0.7, edgecolors="none", label=f'GFN2-xTB, R$^2$={r2_score(x, y):.4f}, RMSE= {root_mean_squared_error(x, y):.3f}')
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def force_component_parity_plot(ref_forces, method_forces, out_path):
    force_ha_bohr_to_mev_a = 1000.0 * 27.211386245988 / 0.529177210903
    ref = np.concatenate([np.asarray(force, dtype=float) for force in ref_forces], axis=0)
    pred = np.concatenate([np.asarray(force, dtype=float) for force in method_forces], axis=0)
    ref = ref * force_ha_bohr_to_mev_a
    pred = pred * force_ha_bohr_to_mev_a

    component_names = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=False, sharey=False)
    for component, ax in enumerate(axes):
        x = ref[:, component]
        y = pred[:, component]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            ax.set_visible(False)
            continue
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.scatter(x, y, s=9, alpha=0.55, edgecolors="none", label="GFN2-xTB, R$^2$={:.4f}, RMSE={:.2f} meV/Å".format(r2_score(x, y), root_mean_squared_error(x, y),))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, linestyle='dashed', alpha=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f'Component {component_names[component]}')
        ax.set_xlabel("Reference force (meV/A)")
        if component == 0:
            ax.set_ylabel("GFN2-xTB force (meV/A)")
        ax.grid(True)
        ax.legend(fontsize=10)
    fig.suptitle(f"Force component parity (RMSE = {root_mean_squared_error(ref, pred):.2f} meV/A)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
