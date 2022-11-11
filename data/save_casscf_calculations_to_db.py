import os
import argparse
import numpy as np
from data.db_utils import xyz_to_db
from ase.db import connect
from typing import List, Tuple

from data.utils import find_all_files_in_output_folder, find_all_geometry_files_in_folder, sort_geometry_files_by_idx

def normalise_rows(mat):
    '''Normalise each row of mat'''
    return np.array(tuple(map(lambda v: v / np.linalg.norm(v), mat)))

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

  geometry_files = sort_geometry_files_by_idx(geometry_files)
  casscf_results = list(sorted(casscf_results, key=lambda x: x.index))
  
  # phase_correct & sort orbitals
  casscf_results[0].mo_coeffs_adjusted = casscf_results[0].mo_coeffs
  casscf_results[0].F_adjusted = casscf_results[0].F
  for casscf_result in casscf_results[1:]:
    casscf_result.mo_coeffs_adjusted = correct_orbitals(ref=casscf_results[0].mo_coeffs.T, 
                                                        target=casscf_result.mo_coeffs.copy().T,
                                                        sort=True).T

  
  # save geometry files & calculated properties
  for idx, geometry_file in enumerate(geometry_files):
    xyz_to_db(geometry_file,
              db_path,
              idx,
              atomic_properties="",
              molecular_properties=[{'mo_coeffs': casscf_results[idx].mo_coeffs.flatten(), 
                                     'mo_coeffs_adjusted': casscf_results[idx].mo_coeffs_adjusted.flatten(), 
                                     'F': casscf_results[idx].F.flatten(),
                                     'mo_energies': casscf_results[idx].mo_energies,
                                     'hcore': casscf_results[idx].h_core.flatten(),
                                     'S': casscf_results[idx].S.flatten(),
                                     'dm': casscf_results[idx].dm.flatten(),
                                     }])

  with connect(db_path) as conn:
    conn.metadata = {"_distance_unit": 'angstrom',
                     "_property_unit_dict": {
                      "mo_coeffs": 1.0, 
                      "mo_coeffs_adjusted": 1.0, 
                      "F": 1.0, 
                      "hcore": 1.0,
                      "S": 1.0,
                      "dm": 1.0
                    },
                     "atomrefs": {
                      'mo_coeffs': [0.0 for _ in range(36)],
                      "mo_coeffs_adjusted": [0.0 for _ in range(36)],
                      'F': [0.0 for _ in range(36)],
                      'hcore': [0.0 for _ in range(36)],
                      'S': [0.0 for _ in range(36)],
                      'dm': [0.0 for _ in range(36)]
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









# def sort_orbitals(ref, target, orb_order):
#   target = target[orb_order]
#   return target

# def get_orbital_order(ref, target):
#     Moverlap=np.dot(normalise_rows(ref), normalise_rows(target).T)
#     orb_order=np.argmax(abs(Moverlap),axis=1)
#     return orb_order

# def get_matrix_indices_to_swap(size: int, index1: int, index2: int) -> List[Tuple[List[int]]]:
#   list = []
#   # 1. swap corners
#   list.append((
#     [index1, index1],
#     [index2, index2]
#   ))
#   list.append((
#     [index1, index2],
#     [index2, index1]
#   ))
#   # 2. swap rows/cols
#   for i in range(size):
#     if i != index1 and i != index2:
#       list.append((
#         [index1, i],
#         [index2, i]
#       ))
#       list.append((
#         [i, index1],
#         [i, index2]
#       ))
#   return list

# def swap_matrix_indices(matrix: np.ndarray, index1: int, index2: int):
#   swap_indices = get_matrix_indices_to_swap(size=matrix.shape[0], index1=index1, index2=index2)
#   for (ind1, ind2) in swap_indices:
#     value1 = matrix[ind1[0], ind1[1]].copy()
#     value2 = matrix[ind2[0], ind2[1]].copy()
#     matrix[ind1[0], ind1[1]] = value2
#     matrix[ind2[0], ind2[1]] = value1
#   return matrix

# def sort_fock_matrix(ref, target, fock):
#   orb_order = get_orbital_order(ref, target)
#   if False in [i in orb_order for i in range(len(orb_order))]:
#     return fock

#   while not np.array_equal(orb_order, np.arange(len(orb_order))):
#     for idx, orb_idx in enumerate(orb_order):
#       if orb_idx != idx:
#         # lets swap the orbitals at orb_idx & idx
#         fock = swap_matrix_indices(fock, index1=idx, index2=orb_idx)
#         # correct orb order
#         orb_order[idx] = orb_order[orb_idx]
#         orb_order[orb_idx] = orb_idx
  
#   return fock