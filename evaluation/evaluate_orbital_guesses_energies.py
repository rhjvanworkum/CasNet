from typing import List, Tuple
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
from pyscf import gto
from pyscf.tools import molden
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from evaluation import initial_guess_dict, compute_casci_energy, compute_casscf_energy, compute_converged_casscf_orbitals


def plot_mo_energies_errors(geometry_files: List[str],
                            method_name: str,
                            model_path: str,
                            basis: str) -> Tuple[np.ndarray, str]:
  mo_e_errors = []
  for geometry_file in geometry_files:
    mo_e_converged, _ = compute_converged_casscf_orbitals(model_path='', 
                                                          geometry_path=geometry_file,
                                                          basis=basis)
    mo_e, _ = initial_guess_dict[method_name](model_path, geometry_file, basis)
    mo_e_errors.append(np.abs(mo_e_converged - mo_e))
  mo_e_errors = np.mean(mo_e_errors, axis=0)
  return mo_e_errors, method_name

def print_casci_energies_errors(geometry_files: List[str], 
                         method_name: str, 
                         model_path: str, 
                         basis: str) -> None:
  errors = []
  for idx, geometry_file in enumerate(geometry_files):
    e_casscf = compute_casscf_energy(geometry_file, basis)
    _, mo = initial_guess_dict[method_name](model_path, geometry_file, basis)
    e_casci = compute_casci_energy(geometry_file, mo , basis)
    print(f'geometry {idx}, error: {np.abs(e_casscf - e_casci)}')
    errors.append(np.abs(e_casscf - e_casci))
  print(f'Method {method_name} CASCI MAE: {np.mean(np.array(errors))} +/- {np.std(np.array(errors))} \n')

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
    if key == 'ao_min':
      print_casci_energies_errors(geometry_files, key, None, basis)
      # print('Calculating orbital energy differences.....\n')
      # mo_e_errors, method_name = plot_mo_energies_errors(geometry_files, key, None, basis)
      # plt.plot(np.arange(len(mo_e_errors)), mo_e_errors, label=method_name)
      # plt.show()
    elif key == 'ML-MO':
      print_casci_energies_errors(geometry_files, key, mo_model, basis)
      # print('Calculating orbital energy differences.....\n')
      # mo_e_errors, method_name = plot_mo_energies_errors(geometry_files, key, mo_model, basis)
      # plt.plot(np.arange(len(mo_e_errors)), mo_e_errors, label=method_name)
      # plt.show()
    elif key == 'ML-F':
      print_casci_energies_errors(geometry_files, key, f_model, basis)
      # print('Calculating orbital energy differences.....\n')
      # mo_e_errors, method_name = plot_mo_energies_errors(geometry_files, key, f_model, basis)
      # plt.plot(np.arange(len(mo_e_errors)), mo_e_errors, label=method_name)
      # plt.show()
    if key == 'phisnet':
      print_casci_energies_errors(geometry_files, key, phisnet_model, basis)
      # print('Calculating orbital energy differences.....\n')
      # mo_e_errors, method_name = plot_mo_energies_errors(geometry_files, key, phisnet_model, basis)
      # plt.plot(np.arange(len(mo_e_errors)), mo_e_errors, label=method_name)
      # plt.show()