
import multiprocessing
import os
from typing import List, Optional
import argparse
import numpy as np
from pyscf import gto, mcscf
from tqdm import tqdm
from functools import reduce
import shutil
import h5py

from data.casscf import EQUILIBRIUM_GEOMETRY_PATH
from data.casscf.openmolcas import MOLCAS_PATH, get_guess_orb_file, get_input_file
from data.utils import CasscfResult, check_and_create_folder, find_all_geometry_files_in_folder, sort_geometry_files_by_distance
from openmolcas.utils import *


def run_fulvene_casscf_calculation(geometry_xyz_file_path: str, 
                                   guess_orb_file_path: str,
                                   base_path: str,
                                   index: int,
                                   basis: str = 'ANO-S-MB'):
    # make dir
    dir_path = f'{base_path}/calculation_{index}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # copy files
    shutil.copy2(get_input_file(basis), f'{dir_path}/CASSCF.input')
    shutil.copy2(guess_orb_file_path, f'{dir_path}/geom.orb')
    shutil.copy2(geometry_xyz_file_path, f'{dir_path}/geom.xyz')

    # create temp dir
    temp_dir = f'{dir_path}/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # execute OpenMolcas
    os.system(f'cd {dir_path} && WorkDir=./temp/ {MOLCAS_PATH} CASSCF.input > calc.log')

    # remove temp dir
    shutil.rmtree(temp_dir)

    # extract information
    data = h5py.File(os.path.join(dir_path, 'CASSCF.rasscf.h5'))
    basis_set_size = int(np.sqrt(data.get('AO_OVERLAP_MATRIX')[:].shape[0]))

    S = data.get('AO_OVERLAP_MATRIX')[:].reshape(basis_set_size, basis_set_size)
    mo_energies = data.get('MO_ENERGIES')[:]
    mo_coeffs = data.get('MO_VECTORS')[:].reshape(basis_set_size, basis_set_size)
    F =  S @ mo_coeffs.T @ np.diag(mo_energies) @ np.linalg.inv(mo_coeffs.T)

    _, _, n_iterations = read_log_file(os.path.join(dir_path, 'calc.log'))
    s1_energy = get_s1_energy(os.path.join(dir_path, 'calc.log'))
    s2_energy = get_s2_energy(os.path.join(dir_path, 'calc.log'))
    e_tot = 0.5 * (s1_energy + s2_energy)

    return CasscfResult(
        converged=True,
        basis=basis,
        e_tot=e_tot,
        mo_energies=mo_energies,
        mo_coeffs=mo_coeffs,
        S=S,
        F=F,
        imacro=n_iterations,
    ), dir_path

def run_casscf_calculations(geometry_folder: str, 
                            output_folder: str,
                            basis: str) -> None:
    check_and_create_folder(geometry_folder)
    check_and_create_folder(output_folder)

    files = find_all_geometry_files_in_folder(geometry_folder)    
    files, _ = sort_geometry_files_by_distance(files, EQUILIBRIUM_GEOMETRY_PATH)   
        
    guess_orb_file = get_guess_orb_file(basis)

    for idx, geometry_file in enumerate(tqdm(files, total=len(files))):
        calculation_name = geometry_file.split('/')[-1].split('.')[0]
        calculation_result, curr_path = run_fulvene_casscf_calculation(geometry_xyz_file_path=geometry_file,
                                                                       guess_orb_file_path=guess_orb_file,
                                                                       base_path=output_folder,
                                                                       index=idx,
                                                                       basis=basis)
                                                                       
        guess_orb_file = os.path.join(curr_path, 'CASSCF.RasOrb')
        calculation_result.store_as_npz(output_folder + calculation_name + '.npz')

    print('Done')

if __name__ == "__main__":
    base_dir = os.environ['base_dir']

    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--basis', type=str)
    args = parser.parse_args()

    run_casscf_calculations(base_dir + args.geometry_folder, base_dir + args.output_folder, args.basis)