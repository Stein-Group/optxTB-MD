import os
from pathlib import Path

from .gradients import calculate_charges_dipole, read_xyz, get_number_of_na_cl_waters
from .process_qchem_charged import extract_final_energy, extract_forces, extract_cm5_charges_qchem
from ase import Atoms



_REPO_ROOT = Path(__file__).resolve().parents[1]


def calculate_re_f_md_dft(structure_file, path=str(_REPO_ROOT / 'structures/cluster_data/dataset/dft_results_revpbe0'), path_monomers=str(_REPO_ROOT / 'structures/cluster_data/monomers/revpbe0'), verbose=True):
    os.chdir(path)
    out_path = os.path.splitext(os.path.basename(structure_file))[0]+'.out'
    energy_self = extract_final_energy(out_path)
    number_of_na, number_of_cl, number_of_waters = get_number_of_na_cl_waters(structure_file)
    #print(f'For a structure {structure_file}, found {number_of_na} Na, {number_of_cl} Cl and {number_of_waters} waters.')
    e_na, e_cl, e_water = 0.0, 0.0, 0.0
    if number_of_na != 0:
        e_na = extract_final_energy(path_monomers + '/na.out')
    if number_of_cl != 0:
        e_cl = extract_final_energy(path_monomers + '/cl.out')
    if number_of_waters != 0:
        e_water = extract_final_energy(path_monomers + '/water.out')
    re = energy_self - number_of_na * e_na - number_of_cl * e_cl - number_of_waters * e_water
    forces = extract_forces(out_path)
    charges = extract_cm5_charges_qchem(out_path)
    coords, species, _ = read_xyz(structure_file)
    if len(charges) != len(coords):
        raise RuntimeError(f"#atoms mismatch: charges={len(charges)} vs xyz={len(coords)}")
    structure_ase = Atoms(symbols=species, positions=coords)
    structure_ase.set_pbc(False)
    _, muDs = calculate_charges_dipole(structure_ase, species, coords, charges, verbose=verbose)

    return re, forces, muDs

