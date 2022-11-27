import argparse
from code import InteractiveInterpreter
from mimetypes import init
import os
from typing import Callable, List, Optional
from phisnet_fork.training.parse_command_line_arguments import parse_command_line_arguments
import numpy as np
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
from pyscf import mcscf, gto

from evaluation import initial_guess_dict, run_casscf_calculation


def evaluate_and_print_initial_guess_convergence(geometry_files: List[str],
                                                 model_path: str, 
                                                 key: str, 
                                                 method: Callable, 
                                                 basis: str):
  macro_iterations = []
  micro_iterations = []
  inner_iterations = []
  e_tots = []
  for idx, geometry_file in enumerate(geometry_files):
    _, mo = method(model_path, geometry_file, basis)
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

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--split_name', type=str)
  parser.add_argument('--mo_model', type=str)
  parser.add_argument('--F_model', type=str)
  parser.add_argument('--phisnet_model', type=str)
  parser.add_argument('--basis', type=str)
  parser.add_argument('--all', type=bool)
  args = parser.parse_args()

  geometry_folder = base_dir + args.geometry_folder
  split_file = './data_storage/' + args.split_name
  mo_model = './checkpoints/' + args.mo_model + '.pt'
  f_model = './checkpoints/' + args.F_model + '.pt'
  phisnet_model = './checkpoints/' + args.phisnet_model + '.pt'
  basis = args.basis

  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  geometry_files = sort_geometry_files_by_idx(geometry_files)
  
  if args.all:
    geometry_files = np.array(geometry_files)
  else:
    geometry_files = np.array(geometry_files)[np.load(split_file)['test_idx']]

  for key, method in initial_guess_dict.items():
    if key == 'ao_min' or 'hartree-fock':
      evaluate_and_print_initial_guess_convergence(geometry_files, None, key, method, basis)
    elif key == 'ML-MO':
      evaluate_and_print_initial_guess_convergence(geometry_files, mo_model, key, method, basis)
    elif key == 'ML-F':
      evaluate_and_print_initial_guess_convergence(geometry_files, f_model, key, method, basis)
    if key == 'phisnet':
      evaluate_and_print_initial_guess_convergence(geometry_files, phisnet_model, key, method, basis)



