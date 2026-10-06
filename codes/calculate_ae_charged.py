import numpy as np
from scipy.optimize import minimize
import re
import os
import subprocess
import matplotlib.pyplot as plt
import toml



def write_custom_toml(data, filepath):
    def write_section(section, prefix=""):
        for key, value in section.items():
            if isinstance(value, dict):
                toml_lines.append(f"[{prefix}{key}]")
                write_section(value, prefix=f"{prefix}{key}.")
            elif isinstance(value, list):
                # Format lists with consistent precision for floats
                value_str = ", ".join(
                    f"{v:.16f}" if isinstance(v, float) else f'"{v}"' if isinstance(v, str) else str(v)
                    for v in value
                )
                toml_lines.append(f"{key} = [ {value_str} ]")
            elif isinstance(value, bool):
                toml_lines.append(f"{key} = {str(value).lower()}")
            else:
                # Ensure version and format are integers if applicable
                if key in ["version", "format"]:
                    toml_lines.append(f"{key} = {value}")
                else:
                    toml_lines.append(
                        f"{key} = {value:.16f}" if isinstance(value, float) else f"{key} = \"{value}\""
                    )

    toml_lines = []
    write_section(data)

    current_line_count = len(toml_lines)
    if current_line_count < 1689:
        # If there are too few lines, pad with empty lines.
        toml_lines.extend([""] * (1689 - current_line_count))
    elif current_line_count > 1689:
        # If there are too many lines, trim the extra ones.
        toml_lines = toml_lines[:1689]

    with open(filepath, "w") as f:
        f.write("\n".join(toml_lines) + "\n")




def generate_parameter_info(param_changes, variation=0.1, one_sided=None):
    """
    Flattens the nested parameter dictionary into a list of parameter names and bounds.
    Each parameter is allowed to vary by ±variation * nominal_value unless overridden
    by `one_sided`, which can enforce one-sided bounds.

    one_sided: dict with keys (elem, key) or (elem, key, idx) and values:
        'up'   -> only allow increase:   [v, v*(1+variation)]
        'down' -> only allow decrease:   [v*(1-variation), v]
    If v == 0, a small range ([-0.2, 0.2]) is used.

    Returns:
      names, bounds, vector, structure
    """
    # Defaults for your example:
    if one_sided is None:
        one_sided = {
            ('Cl', 'zeff'): 'down',  # only decrease from default
            ('Na', 'zeff'): 'up',    # only increase from default
        }

    names = []
    bounds = []
    vector = []
    structure = {}

    for elem, params in param_changes.items():
        structure[elem] = {}
        for key, value in params.items():

            def bound_for(val, rule):
                if val == 0:
                    return [-0.2, 0.2]
                if rule == 'up':
                    # assume nominal val is positive for one-sided 'up' usage
                    return [val, val * (1 + variation)]
                if rule == 'down':
                    return [val * (1 - variation), val]
                # symmetric default
                lower = min(val * (1 - variation), val * (1 + variation))
                upper = max(val * (1 - variation), val * (1 + variation))
                return [lower, upper]

            if isinstance(value, list):
                structure[elem][key] = []
                for i, val in enumerate(value):
                    par_name = f"{elem}_{key}_{i}"
                    names.append(par_name)
                    rule = one_sided.get((elem, key, i), one_sided.get((elem, key)))
                    bound = bound_for(val, rule)
                    bounds.append(bound)
                    vector.append(val)
                    structure[elem][key].append(par_name)
            else:
                par_name = f"{elem}_{key}"
                names.append(par_name)
                rule = one_sided.get((elem, key))
                bound = bound_for(value, rule)
                bounds.append(bound)
                vector.append(value)
                structure[elem][key] = par_name

    return names, bounds, vector, structure

def vector_to_param_changes(vector, structure):
    """
    Convert a sampled vector (list/array of parameter values) into the original
    nested dictionary structure matching that of param_changes.
    Add that lgam[0] is always 1.0 for each element.
    """
    new_params = {}
    # Create an iterator over the vector entries
    vec_iter = iter(vector)
    #print(vec_iter)
    for elem, param_info in structure.items():
        new_params[elem] = {}
        for key, mapping in param_info.items():
            if isinstance(mapping, list):
                # mapping is a list of names corresponding to list parameters
                vals = []
                for name in mapping:
                    if name == 'Fixing to 1.0':
                        # fixed value (lgam[0] = 1.0)
                        vals.append(1.0)
                    elif isinstance(name, tuple) and name[0] == "__fixed__":
                        # fixed value from user specification
                        vals.append(name[1])
                    else:
                        vals.append(next(vec_iter))
                new_params[elem][key] = vals
            else:
                # mapping is a single name
                #print(f'Processing {elem} {key}')
                new_params[elem][key] = next(vec_iter)
    return new_params
