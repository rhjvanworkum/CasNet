import os
import argparse
import numpy as np
from data.db_utils import xyz_to_db
from ase.db import connect
from typing import List, Tuple

from data.utils import find_all_files_in_output_folder, find_all_geometry_files_in_folder, sort_geometry_files_by_idx

def phase_correct_orbitals(ref, target):
  ordered_target = target.copy()

  for idx in range(ordered_target.shape[0]):
      if np.dot(ref[:, idx], ordered_target[:, idx]) < 0:
          ordered_target[:, idx] = -1 * ordered_target[:, idx]

  return ordered_target

def save_casscf_calculations_to_db(geometry_folder: str, output_folder: str, db_path: str) -> None:
  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  casscf_results = find_all_files_in_output_folder(output_folder)
  assert len(geometry_files) == len(casscf_results)

  geometry_files = sort_geometry_files_by_idx(geometry_files)
  casscf_results = list(sorted(casscf_results, key=lambda x: x.index))
  
  # phase_correct & sort orbitals
  casscf_results[0].mo_coeffs_adjusted = casscf_results[0].mo_coeffs
  for idx, casscf_result in enumerate(casscf_results):
    if idx > 0:
      casscf_result.mo_coeffs_adjusted = phase_correct_orbitals(ref=casscf_results[idx - 1].mo_coeffs_adjusted, 
                                                                target=casscf_result.mo_coeffs)

  
  # save geometry files & calculated properties
  for idx, geometry_file in enumerate(geometry_files):
    xyz_to_db(geometry_file,
              db_path,
              idx,
              atomic_properties="",
              molecular_properties=[{'mo_coeffs': casscf_results[idx].mo_coeffs.flatten(), 
                                     'mo_coeffs_adjusted': casscf_results[idx].mo_coeffs_adjusted.flatten(), 
                                     'mo_energies': casscf_results[idx].mo_energies,
                                     'F': casscf_results[idx].F.flatten(),
                                     'S': casscf_results[idx].S.flatten(),
                                     }])

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                     "_property_unit_dict": {
                      "mo_coeffs": 1.0, 
                      "mo_coeffs_adjusted": 1.0, 
                      'mo_energies': 1.0,
                      "F": 1.0, 
                      "S": 1.0,
                    },
                     "atomrefs": {
                      'mo_coeffs': [0.0 for _ in range(36)],
                      "mo_coeffs_adjusted": [0.0 for _ in range(36)],
                      'mo_energies': [0.0 for _ in range(36)],
                      'F': [0.0 for _ in range(36)],
                      'S': [0.0 for _ in range(36)],
                      }
                    }

if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--output_folder', type=str)
  args = parser.parse_args()

  geometry_folder = base_dir + args.geometry_folder
  output_folder = base_dir + args.output_folder
  db_path = './data_storage/' + output_folder.split('/')[-2] + '.db'

  save_casscf_calculations_to_db(geometry_folder, output_folder, db_path)