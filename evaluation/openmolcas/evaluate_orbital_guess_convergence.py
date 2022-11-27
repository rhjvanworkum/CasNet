import argparse
import os
from typing import Callable, List
import numpy as np
import shutil

from data.casscf.openmolcas import MOLCAS_PATH, get_guess_orb_file, get_input_file
from data.casscf.openmolcas.utils import read_log_file, write_coeffs_to_orb_file
from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
from evaluation.openmolcas import initial_guess_dict, basis_dict


def run_casscf_calculation(geometry_xyz_file_path: str, 
                           guess_orbs: np.ndarray,
                           base_path: str,
                           index: int,
                           basis: str = 'ANO-S-MB') -> int:
    # make dir
    dir_path = f'{base_path}/calculation_{index}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # copy files
    shutil.copy2(get_input_file(basis), f'{dir_path}/CASSCF.input')
    shutil.copy2(geometry_xyz_file_path, f'{dir_path}/geom.xyz')
    write_coeffs_to_orb_file(guess_orbs.flatten(), input_file_path=get_guess_orb_file(basis), 
                             output_file_path=f'{dir_path}/geom.orb', n=basis_dict[basis])

    # create temp dir
    temp_dir = f'{dir_path}/temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # execute OpenMolcas
    os.system(f'cd {dir_path} && WorkDir=./temp/ {MOLCAS_PATH} CASSCF.input > calc.log')

    # remove temp dir
    shutil.rmtree(temp_dir)

    return read_log_file(os.path.join(dir_path, 'calc.log'))


def evaluate_and_print_initial_guess_convergence(geometry_files: List[str],
                                                 output_folder: str,
                                                 model_path: str, 
                                                 key: str, 
                                                 method: Callable, 
                                                 basis: str):
  n_its, rasscf_ts, wall_ts = [], [], []
  for idx, geometry_file in enumerate(geometry_files):
    _, mo = method(output_folder, model_path, geometry_file, basis)

    rasscf_t, wall_t, n_it = run_casscf_calculation(geometry_xyz_file_path=geometry_file,
                                                                      guess_orbs=mo,
                                                                      base_path=output_folder,
                                                                      index=idx,
                                                                      basis=basis)
    print(f'{key} at calc {idx}: converged: {n_it} - {wall_t} - {rasscf_t}')
    n_its.append(n_it)
    wall_ts.append(wall_t)
    rasscf_ts.append(rasscf_t)

  print(f'Method {key} convergence: \n Macro iterations: {np.mean(np.array(n_its))} +/- {np.std(np.array(n_its))} \
                                    \n Wall timing: {np.mean(np.array(wall_ts))} +/- {np.std(np.array(wall_ts))} \
                                    \n Rasscf timing: {np.mean(np.array(rasscf_ts))} +/- {np.std(np.array(rasscf_ts))}')


if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--output_folder', type=str)
  parser.add_argument('--split_name', type=str)
  parser.add_argument('--phisnet_model', type=str)
  parser.add_argument('--basis', type=str)
  parser.add_argument('--all', type=bool)
  args = parser.parse_args()

  geometry_folder = base_dir + args.geometry_folder
  output_folder = base_dir + args.output_folder
  split_file = './data_storage/' + args.split_name
  phisnet_model = './checkpoints/' + args.phisnet_model + '.pt'
  basis = args.basis

  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  geometry_files = sort_geometry_files_by_idx(geometry_files)
  
  if args.all:
    geometry_files = np.array(geometry_files)
  else:
    geometry_files = np.array(geometry_files)[np.load(split_file)['test_idx']]

  for key, method in initial_guess_dict.items():
    evaluate_and_print_initial_guess_convergence(geometry_files, output_folder, phisnet_model, key, method, basis)



