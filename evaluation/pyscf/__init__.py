from typing import Tuple
from model.inference import infer_orbitals_from_F_model, infer_orbitals_from_mo_model, infer_orbitals_from_phisnet_model
from pyscf import gto, scf
import numpy as np
import scipy.linalg

basis_dict = {
  'sto_6g': 36,
}

convention = {
    'sto_6g': 'fulvene_minimal_basis'
}

"""
Overlap Matrix
"""

def calculate_overlap_matrix(geometry_path: str, basis: str) -> np.ndarray:
  mol = gto.M(atom=geometry_path,
              basis=basis,
              spin=0)
  myscf = mol.RHF()
  return myscf.get_ovlp(mol)


"""
Fn's to compute orbitals using different methods
"""


def compute_ao_min_orbitals(model_path: str,
                            geometry_path: str,
                            basis: str) -> Tuple[np.ndarray, np.ndarray]:
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  myscf = molecule.RHF()
  guess_dm = scf.hf.init_guess_by_minao(molecule)
  S = myscf.get_ovlp(molecule)
  F = myscf.get_fock(dm=guess_dm)
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo_e, mo
  
def compute_huckel_orbitals(model_path: str,
                            geometry_path: str,
                            basis: str) -> Tuple[np.ndarray, np.ndarray]:
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  myscf = molecule.RHF()
  guess_dm = scf.hf.init_guess_by_huckel(molecule)
  S = myscf.get_ovlp(molecule)
  F = myscf.get_fock(dm=guess_dm)
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo_e, mo

def compute_hf_orbitals(model_path: str,
                        geometry_path: str,
                        basis: str) -> Tuple[np.ndarray, np.ndarray]:
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  hartree_fock = molecule.RHF()
  hartree_fock.kernel()
  return hartree_fock.mo_energy, hartree_fock.mo_coeff

def compute_mo_model_orbitals(model_path: str,
                              geometry_path: str,
                              basis: str):
  basis_set_size = basis_dict[basis]
  mo = infer_orbitals_from_mo_model(model_path, geometry_path, basis_set_size)
  return np.zeros(len(mo)), mo

def compute_F_model_orbitals(model_path: str,
                             geometry_path: str,
                             basis: str):
  basis_set_size = basis_dict[basis]
  F = infer_orbitals_from_F_model(model_path, geometry_path, basis_set_size)
  S = calculate_overlap_matrix(geometry_path, basis)
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo_e, mo

def compute_phisnet_model_orbitals(model_path: str,
                                   geometry_path: str,
                                   basis: str):
  orbital_convention = convention[basis]
  F = infer_orbitals_from_phisnet_model(model_path, geometry_path, orbital_convention)
  S = calculate_overlap_matrix(geometry_path, basis)
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo_e, mo


initial_guess_dict = {
  # 'ao_min': compute_ao_min_orbitals,
  # 'hartree-fock': compute_hf_orbitals,
  # 'ML-MO': compute_mo_model_orbitals,
  # 'ML-F': compute_F_model_orbitals,
  'phisnet': compute_phisnet_model_orbitals
}