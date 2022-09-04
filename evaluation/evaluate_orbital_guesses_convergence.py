import argparse
from code import InteractiveInterpreter
from mimetypes import init
import os
from typing import Callable, List
import numpy as np
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files
from pyscf import mcscf, gto


from evaluation.utils import compute_F_model_orbitals, compute_ao_min_orbitals, compute_mo_model_orbitals

initial_guess_dict = {
  # 'ao_min': compute_ao_min_orbitals,
  'ML-MO': compute_mo_model_orbitals,
  'ML-F': compute_F_model_orbitals,
}

def run_casscf_calculation(geometry_file: str,
                           guess_orbitals: np.ndarray,
                           basis='sto-6g'):
  molecule = gto.M(atom=geometry_file,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0

  hartree_fock = molecule.RHF()
  n_states = 2
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(guess_orbitals)
  return imacro, imicro, iinner

def evaluate_and_print_initial_guess_convergence(geometry_files: List[str],
                                                 model_path: str, 
                                                 key: str, 
                                                 method: Callable, 
                                                 basis: str):
  macro_iterations = []
  micro_iterations = []
  inner_iterations = []
  for idx, geometry_file in enumerate(geometry_files):
    _, mo = method(model_path, geometry_file, basis)
    imacro, imicro, iinner = run_casscf_calculation(geometry_file, mo)
    print(f'{key} at calc {idx}: {imacro} - {imicro} - {iinner}')
    macro_iterations.append(imacro)
    micro_iterations.append(imicro)
    inner_iterations.append(iinner)
  print(f'Method {key} convergence: \n \
        Macro iterations: {np.mean(np.array(macro_iterations))} +/- {np.std(np.array(macro_iterations))} \n \
        Micro iterations: {np.mean(np.array(micro_iterations))} +/- {np.std(np.array(micro_iterations))} \n \
        Inner iterations: {np.mean(np.array(inner_iterations))} +/- {np.std(np.array(inner_iterations))} \n')

if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--split_name', type=str)
  parser.add_argument('--mo_model', type=str)
  parser.add_argument('--F_model', type=str)
  parser.add_argument('--basis', type=str)
  args = parser.parse_args()

  geometry_folder = base_dir + args.geometry_folder
  split_file = './data_storage/' + args.split_name
  mo_model = './checkpoints/' + args.mo_model + '.pt'
  F_model = './checkpoints/' + args.F_model + '.pt'
  basis = args.basis

  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  geometry_files = sort_geometry_files(geometry_files)
  geometry_files = np.array(geometry_files)[np.load(split_file)['val_idx']]

  for key, method in initial_guess_dict.items():
    model_path = ''
    if key == 'ML-MO':
      model_path = mo_model
    elif key == 'ML-F':
      model_path = F_model
    evaluate_and_print_initial_guess_convergence(geometry_files, model_path, key, method, basis)



