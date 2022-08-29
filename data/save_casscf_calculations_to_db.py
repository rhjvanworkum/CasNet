import os
import argparse
import numpy as np
from data.db_utils import xyz_to_db
from ase.db import connect

from data.utils import find_all_files_in_output_folder, find_all_geometry_files_in_folder, sort_geometry_files

def normalise_rows(mat):
    '''Normalise each row of mat'''
    return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

def flip(v):
    '''Returns 1 if max(abs(v))) is positive, and -1 if negative'''
    max_pos = np.argmax(abs(v))
    return v[max_pos] / abs(v[max_pos])

def correct_orbitals(ref, target, sort=False):
    '''Reorder target molecular orbitals according to maximum overlap with ref.
    Orbitals phases are also adjusted to match ref.'''
    if sort:
        Moverlap=np.dot(normalise_rows(ref), normalise_rows(target).T)
        orb_order=np.argmax(abs(Moverlap),axis=1)
        target = target[orb_order]

    for idx in range(target.shape[0]):
        if np.dot(ref[idx, :], target[idx, :]) < 0:
            target[idx, :] = -1 * target[idx, :]

    return target

def save_casscf_calculations_to_db(geometry_folder: str, output_folder: str, db_path: str) -> None:
  geometry_files = find_all_geometry_files_in_folder(geometry_folder)
  casscf_results = find_all_files_in_output_folder(output_folder)
  assert len(geometry_files) == len(casscf_results)

  geometry_files = sort_geometry_files(geometry_files)
  casscf_results = list(sorted(casscf_results, key=lambda x: x.index))
  
  # phase_correct & sort orbitals
  casscf_results[0].mo_coeffs_adjusted = casscf_results[0].mo_coeffs
  for casscf_result in casscf_results[1:]:
    casscf_result.mo_coeffs_adjusted = correct_orbitals(ref=casscf_results[0].mo_coeffs.T, 
                                                        target=casscf_result.mo_coeffs.T).T
  
  # save geometry files & calculated properties
  for idx, geometry_file in enumerate(geometry_files):
    xyz_to_db(geometry_file,
              db_path,
              idx,
              atomic_properties="",
              molecular_properties=[{'mo_coeffs': casscf_results[idx].mo_coeffs.flatten(), 
                                     'mo_coeffs_adjusted': casscf_results[idx].mo_coeffs_adjusted.flatten(), 
                                     'F': casscf_results[idx].F.flatten(),
                                     'S': casscf_results[idx].S.flatten()}])

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                     "_property_unit_dict": {
                      "mo_coeffs": 1.0, 
                      "mo_coeffs_adjusted": 1.0, 
                      "F": 1.0, 
                      "S": 1.0
                    },
                     "atomrefs": {
                      'mo_coeffs': [0.0 for _ in range(36)],
                      "mo_coeffs_adjusted": [0.0 for _ in range(36)],
                      'F': [0.0 for _ in range(36)],
                      'S': [0.0 for _ in range(36)]
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