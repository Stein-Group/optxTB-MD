import os
import math
import numpy as np
import re


def extract_final_energy(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()

    energy = None
    for line in reversed(lines):
        if 'Total energy in the final basis set =' in line:
            energy = float(line.split()[-1])
            break

    if energy is None:
        raise ValueError("Final energy not found in output file.")
    
    return energy

def parse_standard_nuclear_orientation(lines, start_idx):
    """
    Parse the 'Standard Nuclear Orientation (Angstroms)' table starting at start_idx.
    Returns (symbols, coords, end_idx_after_table).
    """
    atoms = []
    coords = []
    i = start_idx

    # 1) Skip down to the first full dashed line beneath the column headers
    dash = re.compile(r'^\s*-{3,}\s*$')
    while i < len(lines) and not dash.match(lines[i]):
        i += 1
    if i >= len(lines):
        return atoms, np.empty((0,3)), i
    i += 1  # move past the first dashed line

    # 2) Read rows until the next dashed line
    row_re = re.compile(
        r'^\s*(\d+)\s+([A-Za-z]{1,2})\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*$'
    )
    while i < len(lines):
        if dash.match(lines[i]):  # reached the closing dashed line
            i += 1
            break
        m = row_re.match(lines[i])
        if m:
            atoms.append(m.group(2))
            coords.append([float(m.group(3)), float(m.group(4)), float(m.group(5))])
        i += 1

    return atoms, np.array(coords, dtype=float), i


def parse_gradient_block(lines, start_idx, natoms):
    # The gradient is printed in blocks of 6 columns:
    # header line with atom indices, then 3 rows (x,y,z) values.
    grads = np.zeros((natoms, 3), dtype=float)
    i = start_idx
    read = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("Max gradient component"):
            break
        # header with atom indices? it will have numbers and no leading '1','2','3'
        cols = line.strip().split()
        is_header = len(cols) > 0 and cols[0].isdigit() and len(cols) >= 1 and line.lstrip().startswith(cols[0])
        # But Q-Chem also prints the row labels '1','2','3' at the start of data rows.
        # We detect headers by peeking next lines to find three rows starting with 1,2,3.
        if is_header and (i+3 < len(lines) and
                          lines[i+1].lstrip().startswith('1') and
                          lines[i+2].lstrip().startswith('2') and
                          lines[i+3].lstrip().startswith('3')):
            # which atom indices are in this block?
            atom_idx = [int(x) for x in cols]
            # read three component rows
            for comp_row in range(3):  # 0->x,1->y,2->z
                vals = lines[i+1+comp_row].split()[1:]  # skip the leading row label
                for k, aidx in enumerate(atom_idx):
                    if aidx-1 < natoms and k < len(vals):
                        grads[aidx-1, comp_row] = float(vals[k])
            read += len(atom_idx)
            i += 4
        else:
            i += 1
    return grads

def extract_forces(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()

    grad_idx = max(i for i, l in enumerate(lines) if "Gradient of SCF Energy" in l)
    if grad_idx is None:
        raise RuntimeError("Could not find 'Gradient of SCF Energy' in the output.")
    sno_idx = next(i for i,l in enumerate(lines) if "Standard Nuclear Orientation (Angstroms)" in l)
    if sno_idx is None:
        raise RuntimeError("Could not find 'Standard Nuclear Orientation' in the output.")
    symbols, coords, _ = parse_standard_nuclear_orientation(lines, sno_idx)
    natoms = len(symbols)
    grads = parse_gradient_block(lines, grad_idx+1, natoms)
    forces = - grads
    return np.array(forces)



def extract_cm5_charges_qchem(output_file):
    """
    Parse 'Charge Model 5' from a Q-Chem-like output.
    Returns a 1D numpy array of charges (floats).
    """
    charges = []
    in_section = False     # after we see the section title
    reading_rows = False   # between the dashed lines that bound the table

    # rows look like: "     1 O                    -0.677896"
    row_re = re.compile(r'^\s*\d+\s+\S+\s+([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*$')

    with open(output_file, "r") as f:
        for line in f:
            s = line.strip()

            # Enter the specific charges section
            if not in_section:
                #print(line)
                if s.startswith("Charge Model 5"):
                    in_section = True
                continue

            # Once in the section, stop at the summary line
            if s.startswith("Sum of atomic charges"):
                break

            # The table is bounded by dashed lines. Toggle reading only within this section.
            if s.startswith("---"):
                reading_rows = not reading_rows
                continue

            if not reading_rows:
                # Skip the "Atom  Charge" header line and any whitespace
                continue

            # Try to parse a data row (index, symbol, charge)
            m = row_re.match(line)
            if m:
                charges.append(float(m.group(1)))
            # else ignore any non-matching line inside the table (rare)

    return np.array(charges)





