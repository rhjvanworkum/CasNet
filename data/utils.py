from operator import itemgetter
from typing import List
import numpy as np
import os

class CasscfResult:
  """
  Base class to store PySCF CASSCF calculation outputs
  """
  def __init__(self, 
               converged: bool,
               basis: str,
               e_tot: float,
               mo_energies: np.ndarray, 
               mo_coeffs: np.ndarray, 
               S: np.ndarray, 
               F: np.ndarray, 
               imacro: int,
               index: int = None) -> None:
    self.converged = converged
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
    np.savez(file, converged=self.converged, basis=self.basis, e_tot=self.e_tot, mo_energies=self.mo_energies, mo_coeffs=self.mo_coeffs, S=self.S, F=self.F, imacro=self.imacro)

  @classmethod
  def load_from_npz(cls, file: str):
    try:
      index = int(file.split('/')[-1].split('.')[0].split('_')[-1])
    except:
      index = None
    
    data = np.load(file, allow_pickle=True)
    return cls(data['converged'], data['basis'], data['e_tot'], 
               data['mo_energies'], data['mo_coeffs'],
               data['S'], data['F'], data['imacro'], index)


def find_all_geometry_files_in_folder(geometry_folder: str) -> List[str]:
  geometry_files = []
  for _, _, files in os.walk(geometry_folder):
    for file in files:
      if '.xyz' in file:
        geometry_files.append(geometry_folder + file)
  return geometry_files


def sort_geometry_files_by_idx(geometry_files: List[str]) -> List[str]:
  files_with_idx = []
  for file in geometry_files:
    files_with_idx.append((int(file.split('/')[-1].split('.')[0].split('_')[-1]), file))
  sorted_files_with_idx = list(sorted(files_with_idx, key=lambda x: x[0]))
  return [file for _, file in sorted_files_with_idx]

def sort_geometry_files_by_distance(geometry_files: List[str], start_geometry_file: str) -> List[str]:
  start_geometry = read_xyz_file(start_geometry_file)
  geometries = [read_xyz_file(file) for file in geometry_files]
  selected_idxs = []
  sorted_geometry_files = []

  current_geometry = start_geometry
  for iteration in range(len(geometry_files)):
    distances = []
    idxs = []

    for idx, geometry in enumerate(geometries):
      if idx not in selected_idxs:
        distances.append(np.sum([np.linalg.norm(atom2.coordinates - atom1.coordinates) for atom1, atom2 in zip(geometry, current_geometry)]))
        idxs.append(idx)

    top_idx = np.argmin(distances)
    selected_idx = idxs[top_idx]

    selected_idxs.append(selected_idx)
    current_geometry = geometries[selected_idx] 
    sorted_geometry_files.append(geometry_files[selected_idx])

  return sorted_geometry_files, selected_idxs

def find_all_files_in_output_folder(output_folder: str) -> List[CasscfResult]:
  file_list = []
  for _, _, files in os.walk(output_folder):
    for file in files:
      file_list.append(output_folder + file)
  return  [CasscfResult.load_from_npz(file) for file in file_list]


def check_and_create_folder(folder):
  if not os.path.exists(folder):
    os.makedirs(folder)

class Atom:
  def __init__(self, type, x, y, z) -> None:
    self.type = type
    self.x = x
    self.y = y
    self.z = z
  
  @property
  def coordinates(self):
    return np.array([self.x, self.y, self.z])

def write_xyz_file(atoms: List[Atom], filename: str):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
      f.write(atom.type)
      for coord in ['x', 'y', 'z']:
        if getattr(atom, coord) < 0:
          f.write('         ')
        else:
          f.write('          ')
        f.write("%.5f" % getattr(atom, coord))
      f.write('\n')
    
    f.write('\n')

def read_xyz_file(filename):
  atoms = []

  with open(filename) as f:
    n_atoms = int(f.readline())
    _ = f.readline()

    for i in range(n_atoms):
      data = f.readline().replace('\n', '').split(' ')
      data = list(filter(lambda a: a != '', data))
      atoms.append(Atom(data[0], float(data[1]), float(data[2]), float(data[3])))

  return atoms