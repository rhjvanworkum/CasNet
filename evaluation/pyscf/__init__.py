from typing import Tuple
from evaluation import compute_cas_orb_energies_from_mo
from model.inference import infer_orbitals_from_F_model, infer_orbitals_from_mo_model, infer_orbitals_from_phisnet_model
from pyscf import gto, scf, mcscf
import numpy as np
import scipy.linalg
import os
import argparse

basis_dict = {
  'sto_6g': 36,
}

convention = {
    'sto_6g': 'fulvene_minimal_basis',
    'cc-pVDZ': 'fulvene_cc-pVDZ'
}

"""
Commmand line argument parsing
"""
def parse_cli_args():
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--split_name', type=str)
  parser.add_argument('--model', type=str)
  parser.add_argument('--mode', type=str)
  parser.add_argument('--basis', type=str)
  parser.add_argument('--all', type=bool)
  args = parser.parse_args()

  geometry_folder = base_dir + args.geometry_folder
  split_file = './data_storage/' + args.split_name
  model_path = './checkpoints/' + args.model + '.pt'
  mode = args.mode
  basis = args.basis
  all = args.all

  return geometry_folder, split_file, model_path, mode, basis, all



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


def compute_ao_min_orbitals(
  model_path: str,
  geometry_path: str,
  basis: str,
  compute_cas_orb_e: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0
  myscf = molecule.RHF()
  guess_dm = scf.hf.init_guess_by_minao(molecule)
  S = myscf.get_ovlp(molecule)
  F = myscf.get_fock(dm=guess_dm)
  mo_e, mo = scipy.linalg.eigh(F, S)

  if compute_cas_orb_e:
    mo_e = compute_cas_orb_energies_from_mo(geometry_path, basis, mo)
  else:
    mo_e = mo_e

  return mo_e, mo
  
def compute_huckel_orbitals(
  model_path: str,
  geometry_path: str,
  basis: str
) -> Tuple[np.ndarray, np.ndarray]:
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0
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
  molecule.verbose = 0
  hartree_fock = molecule.RHF()
  hartree_fock.kernel()
  return hartree_fock.mo_energy, hartree_fock.mo_coeff

def compute_casscf_orbitals(model_path: str,
                            geometry_path: str,
                            basis: str) -> Tuple[np.ndarray, np.ndarray]:
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0

  hartree_fock = molecule.RHF()
  hartree_fock.kernel()

  n_states = 3
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  
  mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
  mo = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo)

  conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeff, mo_e = casscf.kernel(mo)
  return mo_e, mo_coeff

def compute_mo_model_orbitals(
  model_path: str,
  geometry_path: str,
  basis: str,
  compute_cas_orb_e: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
  basis_set_size = basis_dict[basis]
  mo = infer_orbitals_from_mo_model(model_path, geometry_path, basis_set_size)

  if compute_cas_orb_e:
    mo_e = compute_cas_orb_energies_from_mo(geometry_path, basis, mo)
  else:
    mo_e = np.zeros(len(mo))
  return mo_e, mo

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
  'ao_min': compute_ao_min_orbitals,
  'hartree-fock': compute_hf_orbitals,
  'casscf': compute_casscf_orbitals,
  'ML-MO': compute_mo_model_orbitals,
  'ML-F': compute_F_model_orbitals,
  'phisnet': compute_phisnet_model_orbitals,
  'PhiSNet': compute_phisnet_model_orbitals
}