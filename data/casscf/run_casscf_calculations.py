
import multiprocessing
import os
from typing import List, Optional
import argparse
import numpy as np
from pyscf import gto, mcscf
from tqdm import tqdm
from functools import reduce

from data.utils import CasscfResult, check_and_create_folder, find_all_geometry_files_in_folder, sort_geometry_files_by_idx, sort_geometry_files_by_distance

def run_fulvene_casscf_calculation(geometry_xyz_file_path: str, 
                                   basis: str = 'sto_6g',
                                   guess_mos: Optional[np.ndarray] = None):
  molecule = gto.M(atom=geometry_xyz_file_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)

  hartree_fock = molecule.RHF()
  hartree_fock.kernel()
  S = hartree_fock.get_ovlp(molecule)

  n_states = 3
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  
  if not guess_mos is None:
    mo = mcscf.project_init_guess(casscf, guess_mos)
  else: 
    mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
  mo = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo)

  conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)

  F = casscf.get_fock()

  h_core = casscf.get_hcore()

  casdm1 = casscf.fcisolver.make_rdm1(casscf.ci, casscf.ncas, casscf.nelecas)
  mo_coeff = casscf.mo_coeff
  ncore = casscf.ncore
  nocc = casscf.ncore + casscf.ncas
  dm_core = np.dot(mo_coeff[:,:ncore]*2, mo_coeff[:,:ncore].conj().T)
  mocas = mo_coeff[:,ncore:nocc]
  dm = dm_core + reduce(np.dot, (mocas, casdm1, mocas.conj().T))

  return CasscfResult(
    converged=conv,
    basis=basis,
    e_tot=e_tot,
    mo_energies=mo_energies,
    mo_coeffs=mo_coeffs,
    S=S,
    F=F,
    h_core=h_core,
    imacro=imacro,
    dm=dm
  ), mo_coeffs

def run_casscf_calculations(geometry_folder: str, 
                            output_folder: str,
                            basis: str) -> None:
  
  check_and_create_folder(geometry_folder)
  check_and_create_folder(output_folder)

  guess_mos = None

  files = find_all_geometry_files_in_folder(geometry_folder)    

  # files = sort_geometry_files_by_idx(files)
  files = sort_geometry_files_by_distance(files, '/home/ruard/Documents/experiments/fulvene/geometries/geom_scan_200/geometry_0.xyz')   
  print(len(files))  
      
  for file in tqdm(files, total=len(files)):
    calculation_name = file.split('/')[-1].split('.')[0]
    calculation_result, mo_coeffs = run_fulvene_casscf_calculation(file, basis, guess_mos)
    guess_mos = mo_coeffs
    calculation_result.store_as_npz(output_folder + calculation_name + '.npz')
  
  print('Done')

if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--output_folder', type=str)
  parser.add_argument('--basis', type=str)
  args = parser.parse_args()

  run_casscf_calculations(base_dir + args.geometry_folder, base_dir + args.output_folder, args.basis)