from typing import List, Tuple
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files
from evaluation.utils import compute_F_model_orbitals, compute_ao_min_orbitals, compute_casci_energy, compute_converged_casci_energy, compute_converged_casscf_orbitals, compute_huckel_orbitals, compute_mo_model_orbitals
from pyscf import gto
from pyscf.tools import molden
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

initial_guess_dict = {
  'ao_min': compute_ao_min_orbitals,
  'huckel': compute_huckel_orbitals,
  'ML-MO': compute_mo_model_orbitals,
  'ML-F': compute_F_model_orbitals,
}

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
  for geometry_file in geometry_files:
    e_conv = compute_converged_casci_energy(geometry_file, basis)
    _, mo = initial_guess_dict[method_name](model_path, geometry_file, basis)
    e_cas = compute_casci_energy(geometry_file, mo , basis)
    errors.append(np.abs(e_conv - e_cas))
  print(f'Method {method_name} CASCI MAE: {np.mean(np.array(errors))} +/- {np.std(np.array(errors))} \n')

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
    print_casci_energies_errors(geometry_files, key, model_path, basis)
    mo_e_errors, method_name = plot_mo_energies_errors(geometry_files, key, model_path, basis)
    plt.plot(np.arange(len(mo_e_errors)), mo_e_errors, label=method_name)
  
  plt.savefig(f'results/{F_model}-{mo_model}_mo_e.png')

