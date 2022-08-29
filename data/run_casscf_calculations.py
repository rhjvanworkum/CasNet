
import multiprocessing
import os
from typing import List
import argparse
import numpy as np
from pyscf import gto, mcscf
from tqdm import tqdm

from utils import CasscfResult, check_and_create_folder, find_all_geometry_files_in_folder

def run_fulvene_casscf_calculation(geometry_xyz_file_path: str, 
                                   basis: str = 'sto_6g') -> CasscfResult:
  molecule = gto.M(atom=geometry_xyz_file_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)

  hartree_fock = molecule.RHF()
  hartree_fock.kernel()
  S = hartree_fock.get_ovlp(molecule)

  n_states = 2
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
  mo = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo)

  e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)
  F = casscf.get_fock()

  return CasscfResult(
    basis=basis,
    e_tot=e_tot,
    mo_energies=mo_energies,
    mo_coeffs=mo_coeffs,
    S=S,
    F=F,
    imacro=imacro
  )

def run_casscf_calculations_parallel(geometry_folder: str, 
                                     output_folder: str,
                                     basis: str) -> None:
  check_and_create_folder(geometry_folder)
  check_and_create_folder(output_folder)

  def job(file, output_folder, basis) -> None:
    calculation_name = file.split('.')[0]
    calculation_result = run_fulvene_casscf_calculation(file, basis)
    calculation_result.store_as_npz(output_folder + calculation_name + '.npz')
    return None

  files = find_all_geometry_files_in_folder(geometry_folder)
  parallel_args = [(file, output_folder, basis) for file in files]

  pool = multiprocessing.Pool(N_JOBS)
  for result in tqdm(pool.imap(job, parallel_args), total=len(parallel_args)):
    pass

  print('Done')
                      

def run_casscf_calculations(geometry_folder: str, 
                            output_folder: str,
                            basis: str) -> None:
  
  check_and_create_folder(geometry_folder)
  check_and_create_folder(output_folder)

  files = find_all_geometry_files_in_folder(geometry_folder)              
  for file in tqdm(files, total=len(files)):
    calculation_name = file.split('/')[-1].split('.')[0]
    calculation_result = run_fulvene_casscf_calculation(file, basis)
    calculation_result.store_as_npz(output_folder + calculation_name + '.npz')
  
  print('Done')

if __name__ == "__main__":
  N_JOBS = 4
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--output_folder', type=str)
  parser.add_argument('--basis', type=str)
  parser.add_argument('--parallel', action='store_true')
  parser.add_argument('--no-parallel', dest='parallel', action='store_false')
  parser.set_defaults(parallel=True)
  args = parser.parse_args()

  if args.parallel:
    run_casscf_calculations_parallel(base_dir + args.geometry_folder, base_dir + args.output_folder, args.basis)
  else:
    run_casscf_calculations(base_dir + args.geometry_folder, base_dir + args.output_folder, args.basis)