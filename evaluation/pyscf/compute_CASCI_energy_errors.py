from typing import List, Tuple, Callable
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
import numpy as np
import matplotlib.pyplot as plt

from evaluation import compute_casci_energy, compute_casscf_energy, compute_converged_casscf_orbitals
from evaluation.pyscf import initial_guess_dict, parse_cli_args

def plot_mo_energies_errors(
  geometry_files: List[str],
  method: Callable,
  model_path: str,
  basis: str
) -> np.ndarray:
  mo_e_errors = []
  for geometry_file in geometry_files:
    mo_e_converged, _ = compute_converged_casscf_orbitals(model_path='', 
                                                          geometry_path=geometry_file,
                                                          basis=basis)
    mo_e, _ = method(model_path, geometry_file, basis)
    mo_e_errors.append(np.abs(mo_e_converged - mo_e))
  mo_e_errors = np.mean(mo_e_errors, axis=0)
  return mo_e_errors

def print_casci_energies_errors(
  geometry_files: List[str], 
  method: Callable, 
  model_path: str, 
  basis: str
) -> None:
  errors = []
  for idx, geometry_file in enumerate(geometry_files):
    e_casscf = compute_casscf_energy(geometry_file, basis)
    _, mo = method(model_path, geometry_file, basis)
    e_casci = compute_casci_energy(geometry_file, mo , basis)
    print(f'geometry {idx}, error: {np.abs(e_casscf - e_casci)}')
    errors.append(np.abs(e_casscf - e_casci))
  print(f'CASCI MAE: {np.mean(np.array(errors))} +/- {np.std(np.array(errors))} \n')

if __name__ == "__main__":
  geometry_folder, split_file, model_path, mode, basis, all = parse_cli_args()

  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  geometry_files = sort_geometry_files_by_idx(geometry_files)
  if all:
    geometry_files = np.array(geometry_files)
  else:
    geometry_files = np.array(geometry_files)[np.load(split_file)['test_idx']]

  method = initial_guess_dict[mode]
  print_casci_energies_errors(geometry_files, method, model_path, basis)