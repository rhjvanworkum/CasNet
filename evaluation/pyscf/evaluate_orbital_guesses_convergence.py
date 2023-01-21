from typing import Callable, List, Optional
import numpy as np

from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
from evaluation.pyscf import initial_guess_dict, parse_cli_args
from evaluation import run_casscf_calculation


def evaluate_and_print_initial_guess_convergence(
  geometry_files: List[str],
  model_path: str, 
  key: str, 
  method: Callable, 
  basis: str
) -> None:
  macro_iterations = []
  micro_iterations = []
  inner_iterations = []
  e_tots = []
  for idx, geometry_file in enumerate(geometry_files):
    _, mo = method(model_path, geometry_file, basis)
    conv, e_tot, imacro, imicro, iinner = run_casscf_calculation(geometry_file, mo, basis=basis)
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
  geometry_folder, split_file, model_path, mode, basis, all = parse_cli_args()

  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  geometry_files = sort_geometry_files_by_idx(geometry_files)
  
  if all:
    geometry_files = np.array(geometry_files)
  else:
    geometry_files = np.array(geometry_files)[np.load(split_file)['test_idx']]

  method = initial_guess_dict[mode]
  evaluate_and_print_initial_guess_convergence(geometry_files, model_path, mode, method, basis)