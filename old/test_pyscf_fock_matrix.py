import os
from typing import List
import argparse
import numpy as np
from data.save_casscf_calculations_to_db import get_orbital_order
from pyscf import gto, mcscf, scf
from functools import reduce
from ase.db import connect
import scipy
import scipy.linalg
import torch
import schnetpack as spk
from ase import io

from test import write_db_entry_to_molden_file

"""
use this function 'def make_env()'

._atm property => presents atoms
  - row 1 => CHARGE      (just atomic number)
  - row 2 => PTR_COORD   () 
  - row 3 => NUC_MOD_OF (nuclear mode either 'nuc_point' or 'nu_gauss')

"""
bohr_to_angstrom = 0.529177249

def extract_coordinates_from_table(table):
  coords = []
  table = table.reshape(6, 4)
  for i in range(6):
    coords.append(table[i, :3] * bohr_to_angstrom)
  return np.round_(np.array(coords), decimals=5)

def read_coords_from_xyz(file):
  coords = []
  with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines[2:8]:
      coords.append([float(num) for num in line.split()[1:]])
  return np.array(coords)

if __name__ == "__main__":
  geometry_file_path = "/mnt/c/users/rhjva/imperial/fulvene/geometries/ethene_geom_scan/geometry_180.xyz"
  molecule = gto.M(atom=geometry_file_path,
                  basis='sto_6g',
                  spin=0,
                  symmetry=True)

  

  hartree_fock = molecule.RHF()
  # hartree_fock.kernel()
  S = hartree_fock.get_ovlp(molecule)

  n_states = 2
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=2, nelecas=2).state_average(weights)
  casscf.conv_tol = 1e-8

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  atoms = io.read(geometry_file_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device)
  input = converter(atoms)
  model = torch.load('checkpoints/ethene_dm_test.pt', map_location=device).to(device)
  model.eval()
  output = model(input)
  values = output['dm'].detach().cpu().numpy()[0]
  dm = values.reshape(14, 14)
  dm = np.asarray(dm.tolist(), order='C')

  def get_fock_from_dm(casscf, dm):
    vj, vk = casscf._scf.get_jk(casscf.mol, dm)
    fock = casscf.get_hcore() + vj-vk*.5
    return fock

  F = get_fock_from_dm(casscf, dm)
  mo_e, mo_coeffs = scipy.linalg.eigh(F, S)
  # mo = mcscf.project_init_guess(casscf, mo_coeffs)
  # mo = casscf.sort_mo([8, 9], mo)
  conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo_coeffs)
  print(imacro, imicro, iinner)

  guess_dm = scf.hf.init_guess_by_minao(molecule)
  F = get_fock_from_dm(casscf, guess_dm)
  mo_e, mo = scipy.linalg.eigh(F, S)
  conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)
  print(imacro, imicro, iinner)

  # mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
  # mo = casscf.sort_mo([8, 9], mo)
  # conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)
  # print(imacro, imicro, iinner)

  # casdm1 = casscf.fcisolver.make_rdm1(casscf.ci, casscf.ncas, casscf.nelecas)
  # mo_coeff = casscf.mo_coeff
  # ncore = casscf.ncore
  # nocc = casscf.ncore + casscf.ncas
  # dm_core = np.dot(mo_coeff[:,:ncore]*2, mo_coeff[:,:ncore].conj().T)
  # mocas = mo_coeff[:,ncore:nocc]
  # dm = dm_core + reduce(np.dot, (mocas, casdm1, mocas.conj().T))

  # mo = mcscf.project_init_guess(casscf, casscf.mo_coeff)
  # mo = casscf.sort_mo([8, 9], mo)
  # conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)
  # print(imacro, imicro, iinner)

  # F = get_fock_from_dm(casscf, dm)
  # mo_e, mo_coeffs = scipy.linalg.eigh(F, S)
  # mo = mcscf.project_init_guess(casscf, mo_coeffs)
  # mo = casscf.sort_mo([8, 9], mo)
  # conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)
  # print(imacro, imicro, iinner)
  
  # with connect('./data_storage/ethene_geom_scan.db') as conn:
  #   row = conn.get(1)
  #   eigvals, mo_coeffs = np.linalg.eig(dm)
  #   write_db_entry_to_molden_file('test_1.molden', row, casscf.mo_coeff)




  # # S = hartree_fock.get_ovlp(molecule)

  # # print('querying fock')
  # # F = hartree_fock.get_fock()

  # # lets start with hcore
  # hcore = hartree_fock.get_hcore(molecule)
  # # kin integral
  # m1 = molecule.intor_symmetric('int1e_kin')
  # m2 = molecule.intor('int1e_kin', hermi=1, aosym='s4')
  # kin_integral = gto.getints('int1e_kin_sph', molecule._atm, molecule._bas, molecule._env, hermi=1, aosym='s4')
  # assert np.array_equal(m1, kin_integral)
  # assert np.array_equal(m2, kin_integral)
  # # nuc integral
  # m1 = molecule.intor_symmetric('int1e_nuc')
  # m2 = molecule.intor('int1e_nuc', hermi=1, aosym='s4')
  # nuc_integral = gto.getints('int1e_nuc_sph', molecule._atm, molecule._bas, molecule._env, hermi=1, aosym='s4')
  # assert np.array_equal(m1, nuc_integral)
  # assert np.array_equal(m2, nuc_integral)
  # # hcore integral
  # assert np.array_equal(kin_integral + nuc_integral, hcore)