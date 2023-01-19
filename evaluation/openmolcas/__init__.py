import os
import shutil
import numpy as np
import h5py
import torch
from typing import Tuple
import scipy
import scipy.linalg

from model.inference import infer_orbitals_from_phisnet_model, infer_orbitals_from_F_model
from data.utils import read_xyz_file
from phisnet_fork.utils.transform_hamiltonians import transform_hamiltonians_from_lm_to_ao
from data.casscf.openmolcas import get_seward_input_file, MOLCAS_PATH
# from openmolcas.utils import *

basis_dict = {
    'ANO-S-MB': 36,
    'cc-pVDZ': 114
}
convention = {
    'ANO-S-MB': 'fulvene_minimal_basis',
    'cc-pVDZ': 'fulvene_cc-pVDZ'
}

"""
Overlap matrix
"""

def calculate_overlap_matrix(base_path, geometry_xyz_file_path: str, basis: str) -> np.ndarray:
    # make dir
    dir_path = f'{base_path}/temporary/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # copy files
    shutil.copy2(get_seward_input_file(basis), f'{dir_path}/CASSCF.input')
    shutil.copy2(geometry_xyz_file_path, f'{dir_path}/geom.xyz')

    # create temp dir
    temp_dir = f'{dir_path}/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # execute OpenMolcas
    os.system(f'cd {dir_path} && WorkDir=./temp/ {MOLCAS_PATH} CASSCF.input > calc.log')

    # extract overlap matrix
    data = h5py.File(os.path.join(dir_path, 'CASSCF.guessorb.h5'))
    basis_set_size = int(np.sqrt(data.get('AO_OVERLAP_MATRIX')[:].shape[0]))

    S = data.get('AO_OVERLAP_MATRIX')[:].reshape(basis_set_size, basis_set_size)

    # remove dir
    shutil.rmtree(dir_path)

    return S

"""
Fn's to compute orbitals using different methods
"""

def compute_huckel_orbitals(base_path: str,
                            model_path: str, 
                            geometry_path: str,
                            basis: str = 'sto_6g') -> Tuple[np.ndarray, np.ndarray]:
    # make dir
    dir_path = f'{base_path}/temporary/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # copy files
    shutil.copy2(get_seward_input_file(basis), f'{dir_path}/CASSCF.input')
    shutil.copy2(geometry_path, f'{dir_path}/geom.xyz')

    # create temp dir
    temp_dir = f'{dir_path}/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # execute OpenMolcas
    os.system(f'cd {dir_path} && WorkDir=./temp/ {MOLCAS_PATH} CASSCF.input > calc.log')

    # extract overlap matrix
    data = h5py.File(os.path.join(dir_path, 'CASSCF.guessorb.h5'))
    basis_set_size = int(np.sqrt(data.get('AO_OVERLAP_MATRIX')[:].shape[0]))

    mo = data.get('MO_VECTORS')[:].reshape(basis_set_size, basis_set_size)
    mo_e = data.get('MO_ENERGIES')[:]

    # remove dir
    shutil.rmtree(dir_path)

    return mo_e, mo

def compute_f_model_orbitals(base_path: str,
                                   model_path: str, 
                                   geometry_path: str,
                                   basis: str = 'sto_6g') -> Tuple[np.ndarray, np.ndarray]:
    F = infer_orbitals_from_F_model(model_path, geometry_path, basis_dict[basis])
    S = calculate_overlap_matrix(base_path, geometry_path, basis)
    mo_e, mo = scipy.linalg.eigh(F, S, driver='gvd')
    return mo_e, mo.T

def compute_phisnet_model_orbitals(base_path: str,
                                   model_path: str, 
                                   geometry_path: str,
                                   basis: str = 'sto_6g') -> Tuple[np.ndarray, np.ndarray]:
    orbital_convention = convention[basis]
    F = infer_orbitals_from_phisnet_model(model_path, geometry_path, orbital_convention)
    S = calculate_overlap_matrix(base_path, geometry_path, basis)
    mo_e, mo = scipy.linalg.eigh(F, S, driver='gvd')
    return mo_e, mo.T

initial_guess_dict = {
#   'huckel': compute_huckel_orbitals,
#   'f_model': compute_f_model_orbitals,
  'phisnet': compute_phisnet_model_orbitals
}