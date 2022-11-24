import argparse
from code import InteractiveInterpreter
from mimetypes import init
import os
from typing import Callable, List, Optional
from phisnet_fork.training.parse_command_line_arguments import parse_command_line_arguments
import numpy as np
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
from pyscf import mcscf, gto


from evaluation.utils import compute_F_model_orbitals, compute_ao_min_orbitals, compute_mo_model_orbitals, compute_phisnet_model_orbitals

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

  conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(guess_orbitals)
  return conv, e_tot, imacro, imicro, iinner

def evaluate_and_print_initial_guess_convergence(geometry_files: List[str],
                                                 model_path: str, 
                                                 key: str, 
                                                 method: Callable, 
                                                 basis: str,
                                                 args: Optional[argparse.Namespace]):
  macro_iterations = []
  micro_iterations = []
  inner_iterations = []
  e_tots = []
  for idx, geometry_file in enumerate(geometry_files):
    _, mo = method(model_path, geometry_file, basis, args)
    conv, e_tot, imacro, imicro, iinner = run_casscf_calculation(geometry_file, mo)
    print(f'{key} at calc {idx}: converged: {conv} {imacro} - {imicro} - {iinner}')
    macro_iterations.append(imacro)
    micro_iterations.append(imicro)
    inner_iterations.append(iinner)
    e_tots.append(e_tot)
  print(f'Method {key} convergence: \n \
        Macro iterations: {np.mean(np.array(macro_iterations))} +/- {np.std(np.array(macro_iterations))} \n \
        Micro iterations: {np.mean(np.array(micro_iterations))} +/- {np.std(np.array(micro_iterations))} \n \
        Inner iterations: {np.mean(np.array(inner_iterations))} +/- {np.std(np.array(inner_iterations))} \n')
  print(e_tots)

if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  initial_guess_dict = {
    'ao_min': compute_ao_min_orbitals,
    'ML-MO': compute_mo_model_orbitals,
    'ML-F': compute_F_model_orbitals,
    'phisnet': compute_phisnet_model_orbitals
  }

  phisnet_args = parse_command_line_arguments()

  geometry_folder = base_dir + 'geometries/fulvene_geom_scan_250/'
  split_file = './data_storage/' + 'fulvene_gs_250_inter.npz'
  basis = 'sto_6g'

  MO_model = './checkpoints/' + 'fulvene_gs250_inter_MO' + '.pt'
  F_model = './checkpoints/' + 'fulvene_gs250_inter_F' + '.pt'
  phisnet_model = './checkpoints/' + 'fulvene_gs250_inter_phisnet' + '.pt'

  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  geometry_files = sort_geometry_files_by_idx(geometry_files)
  geometry_files = np.array(geometry_files)[np.load(split_file)['val_idx']]

  for key, method in initial_guess_dict.items():
    if key == 'ao-min':
      evaluate_and_print_initial_guess_convergence(geometry_files, None, key, method, basis, args=None)
    elif key == 'ML-MO':
      evaluate_and_print_initial_guess_convergence(geometry_files, MO_model, key, method, basis, args=None)
    elif key == 'ML-F':
      evaluate_and_print_initial_guess_convergence(geometry_files, F_model, key, method, basis, args=None)
    if key == 'phisnet':
      evaluate_and_print_initial_guess_convergence(geometry_files, phisnet_model, key, method, basis, args=phisnet_args)



