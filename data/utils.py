from typing import List
import numpy as np
import os

class CasscfResult:
  """
  Base class to store PySCF CASSCF calculation outputs
  """
  def __init__(self, 
               basis: str,
               e_tot: float,
               mo_energies: np.ndarray, 
               mo_coeffs: np.ndarray, 
               S: np.ndarray, 
               F: np.ndarray, 
               imacro: int,
               index: int = None) -> None:
    self.basis = basis
    self.e_tot = e_tot
    self.mo_energies = mo_energies
    self.mo_coeffs = mo_coeffs
    self.S = S
    self.F = F
    self.imacro = imacro

    self.index = index
    self.mo_coeffs_adjusted = None

  def store_as_npz(self, file: str):
    np.savez(file, basis=self.basis, e_tot=self.e_tot, mo_energies=self.mo_energies, mo_coeffs=self.mo_coeffs, S=self.S, F=self.F, imacro=self.imacro)

  @classmethod
  def load_from_npz(cls, file: str):
    try:
      index = int(file.split('/')[-1].split('.')[0].split('_')[-1])
    except:
      index = None
    
    data = np.load(file)
    return cls(data['basis'], data['e_tot'], data['mo_energies'], data['mo_coeffs'],
               data['S'], data['F'], data['imacro'], index)


def find_all_geometry_files_in_folder(geometry_folder: str) -> List[str]:
  geometry_files = []
  for _, _, files in os.walk(geometry_folder):
    for file in files:
      if '.xyz' in file:
        geometry_files.append(geometry_folder + file)
  return geometry_files


def sort_geometry_files(geometry_files: List[str]) -> List[str]:
  files_with_idx = []
  for file in geometry_files:
    files_with_idx.append((int(file.split('/')[-1].split('.')[0].split('_')[-1]), file))
  sorted_files_with_idx = list(sorted(files_with_idx, key=lambda x: x[0]))
  return [file for _, file in sorted_files_with_idx]


def find_all_files_in_output_folder(output_folder: str) -> List[CasscfResult]:
  file_list = []
  for _, _, files in os.walk(output_folder):
    for file in files:
      file_list.append(output_folder + file)
  return  [CasscfResult.load_from_npz(file) for file in file_list]


def check_and_create_folder(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)
