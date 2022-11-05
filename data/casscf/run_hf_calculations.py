
import multiprocessing
import os
from typing import List
import argparse
import numpy as np
from pyscf import gto, mcscf, scf
from tqdm import tqdm

from data.utils import CasscfResult, check_and_create_folder, find_all_geometry_files_in_folder

#
# Loop for optimizing orbitals until stable
#
from pyscf.lib import logger

instab_counter = 0

def stable_opt_internal(mf):
    global instab_counter
    log = logger.new_logger(mf)
    mo1, _, stable, _ = mf.stability(return_status=True)
    cyc = 0
    while (not stable and cyc < 10):
        log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc += 1
    if not stable:
        log.note('Stability Opt failed after %d attempts' % cyc)
        instab_counter += 1
    return mf

def run_fulvene_hf_calculation(geometry_xyz_file_path: str, 
                               basis: str = 'sto_6g') -> CasscfResult:
  global instab_counter
  molecule = gto.M(atom=geometry_xyz_file_path,
                   basis=basis,
                   spin=0,
                   symmetry=True)

  hartree_fock = scf.RHF(molecule).newton().run()
  # log = logger.new_logger(hartree_fock)

  # # internal stablility 
  # mo1, _, stable, _ = hartree_fock.stability(return_status=True)
  # cyc = 0
  # while (not stable and cyc < 10):
  #   log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
  #   dm1 = hartree_fock.make_rdm1(mo1, hartree_fock.mo_occ)
  #   hartree_fock = hartree_fock.run(dm1)
  #   mo1, _, stable, _ = hartree_fock.stability(return_status=True)
  #   cyc += 1
  # if not stable:
  #   log.note('Stability Opt failed after %d attempts' % cyc)
  #   instab_counter += 1

  # # external stabillity
  # _, mo1, _, stable = hartree_fock.stability(return_status=True)
  # cyc = 0
  # while (not stable and cyc < 10):
  #   log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
  #   dm1 = hartree_fock.make_rdm1(mo1, hartree_fock.mo_occ)
  #   hartree_fock = hartree_fock.run(dm1)
  #   _, mo1, _, stable = hartree_fock.stability(return_status=True)
  #   cyc += 1
  # if not stable:
  #   log.note('Stability Opt failed after %d attempts' % cyc)
    # instab_counter += 1


  e_tot = hartree_fock.e_tot
  S = hartree_fock.get_ovlp(molecule)
  F = hartree_fock.get_fock()
  print(hartree_fock.converged)

  return CasscfResult(
    converged=hartree_fock.converged,
    basis=basis,
    e_tot=e_tot,
    mo_energies=hartree_fock.mo_energy,
    mo_coeffs=hartree_fock.mo_coeff,
    S=S,
    F=F,
    imacro=0
  )

def run_casscf_calculations(geometry_folder: str, 
                            output_folder: str,
                            basis: str) -> None:
  global instab_counter
  check_and_create_folder(geometry_folder)
  check_and_create_folder(output_folder)


  files = find_all_geometry_files_in_folder(geometry_folder)              
  for idx, file in tqdm(enumerate(files), total=len(files)):
    calculation_name = file.split('/')[-1].split('.')[0]
    calculation_result = run_fulvene_hf_calculation(file, basis)
    calculation_result.store_as_npz(output_folder + calculation_name + '.npz')
  
  print(instab_counter)
  print('Done')

if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_folder', type=str)
  parser.add_argument('--output_folder', type=str)
  parser.add_argument('--basis', type=str)
  args = parser.parse_args()

  run_casscf_calculations(base_dir + args.geometry_folder, base_dir + args.output_folder, args.basis)