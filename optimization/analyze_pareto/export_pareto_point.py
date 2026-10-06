"""
Write the GFN2-xTB parameter file (TOML) for one point of an NSGA Pareto front.

This is the step between the optimization stages: the TOML of the selected
water point is the FIXED_HO_TOML_PATH of the following Na stage. It applies the
same operations as save_param_toml() in the analyze_pareto notebooks:
fixed H/O parameters of the run (Na stages only) + optimized parameters of the
selected point, written on top of gfn2-xtb.toml with ensure_param_file().

Usage (from the repository root):

  python optimization/analyze_pareto/export_pareto_point.py \
      optimization/T1_partial_charge_schemes/hpc_runs/<RUN_ID>/nsga3_results_t1_EF_param_reg_<RUN_ID>.npz \
      --index 249 --output parameter_files/<name>.toml

The run_config.json of the run is read from the directory of the results file
(or from --run-config). --index refers to the rows of X_pareto (results file)
or X (generation_fronts/*.npz).
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import toml

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from codes.common import load_fixed_params_from_toml, merge_param_changes
from codes.gradients import ensure_param_file


def vector_to_changes(parameter_names, vector, base):
    """Rebuild the nested parameter dict from parameter names such as
    'kpair_O-H', 'H_zeff' or 'Na_slater_0'. List components that are not
    optimized keep their values from the base parameter file."""
    changes = {}
    for name, value in zip(parameter_names, vector):
        elem, key = name.split("_", 1)
        if elem == "kpair":
            changes.setdefault("kpair", {})[key] = float(value)
            continue
        idx = None
        head, _, tail = key.rpartition("_")
        if head and tail.isdigit():
            key, idx = head, int(tail)
        if idx is None:
            changes.setdefault(elem, {})[key] = float(value)
        else:
            values = changes.setdefault(elem, {}).setdefault(key, list(base["element"][elem][key]))
            values[idx] = float(value)
    return changes


def main():
    parser = argparse.ArgumentParser(description="Export one Pareto point of an NSGA run to a GFN2-xTB TOML file.")
    parser.add_argument("results", help="NSGA results .npz (X_pareto) or generation_fronts/*.npz (X)")
    parser.add_argument("--index", type=int, required=True, help="row of the Pareto front to export")
    parser.add_argument("--output", required=True, help="output TOML path")
    parser.add_argument("--run-config", default=None, help="run_config.json (default: next to the results file)")
    parser.add_argument("--base-param-file", default=str(PROJECT_ROOT_PATH / "gfn2-xtb.toml"))
    args = parser.parse_args()

    results = np.load(args.results, allow_pickle=True)
    X = results["X_pareto"] if "X_pareto" in results.files else results["X"]
    parameter_names = [str(n) for n in results["parameter_names"]]
    vector = np.asarray(X[args.index], dtype=float)

    run_config_path = Path(args.run_config) if args.run_config else Path(args.results).resolve().parent / "run_config.json"
    if not run_config_path.exists():
        run_config_path = Path(args.results).resolve().parents[1] / "run_config.json"  # generation_fronts/*.npz
    run_config = json.loads(run_config_path.read_text())

    base = toml.load(args.base_param_file)
    optimized_changes = vector_to_changes(parameter_names, vector, base)

    # Na stages: H/O parameters were fixed to the water stage result
    if run_config.get("fixed_param_changes"):
        param_dict = merge_param_changes(run_config["fixed_param_changes"], optimized_changes)
    elif run_config.get("fixed_ho_toml_path"):
        fixed = load_fixed_params_from_toml(
            run_config["fixed_ho_toml_path"],
            {k: tuple(v) for k, v in run_config["fixed_element_params"].items()},
            tuple(run_config["fixed_kpair_keys"]),
        )
        param_dict = merge_param_changes(fixed, optimized_changes)
    else:
        param_dict = optimized_changes

    with tempfile.TemporaryDirectory() as tmp:
        written = ensure_param_file(param_dict, param_file=args.base_param_file, DIR=tmp)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(written, args.output)

    print(f"Point {args.index} of {args.results}")
    for name, value in zip(parameter_names, vector):
        print(f"  {name:16s} {value:.10f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
