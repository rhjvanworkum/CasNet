from typing import Tuple, Optional, Any
from model.inference import infer_orbitals_from_F_model, infer_orbitals_from_mo_model, infer_orbitals_from_phisnet_model
from pyscf import gto, scf, mcscf
import numpy as np
import scipy.linalg

basis_dict = {
  'sto_6g': 36,
}

N_STATES = 3

def run_casscf_calculation(geometry_file: str,
                           guess_orbitals: np.ndarray,
                           basis='sto-6g'):
  molecule = gto.M(atom=geometry_file,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0

  hartree_fock = molecule.RHF()
  n_states = N_STATES
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  conv, e_tot, imacro, imicro, iinner, _, _, _, _ = casscf.kernel(guess_orbitals)
  return conv, e_tot, imacro, imicro, iinner


def compute_converged_casscf_orbitals(model_path: str,
                                      geometry_path: str,
                                      basis: str):
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0

  hartree_fock = molecule.RHF()
  hartree_fock.kernel()
  S = hartree_fock.get_ovlp(molecule)

  n_states = N_STATES
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
  mo = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo)

  _, _, _, _, _, _, _, mo_coeffs, mo_energies = casscf.kernel(mo)
  return mo_energies, mo_coeffs


def compute_casci_energy(geometry_path: str,
                         orbitals: np.ndarray,
                         basis: str) -> float: 
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0

  hartree_fock = molecule.RHF()

  n_states = N_STATES
  weights = np.ones(n_states) / n_states
  casci = hartree_fock.CASCI(ncas=6, nelecas=6).state_average(weights)
  casci.conv_tol = 1e-8

  output = casci.kernel(orbitals)
  return output[0]


def compute_casscf_energy(geometry_path: str,
                                   basis: str) -> float: 
  molecule = gto.M(atom=geometry_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)
  molecule.verbose = 0

  hartree_fock = molecule.RHF()
  hartree_fock.kernel()
  S = hartree_fock.get_ovlp(molecule)

  n_states = N_STATES
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
  mo = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo)

  _, e_tot, _, _, _, _, _, _, _ = casscf.kernel(mo)
  return e_tot