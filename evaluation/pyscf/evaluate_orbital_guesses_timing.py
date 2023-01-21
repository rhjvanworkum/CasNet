"""
Unused for now
"""

# import os
# import time
# import numpy as np
# from pyscf import mcscf, gto
# import multiprocessing
# from tqdm import tqdm

# from evaluation.utils import compute_F_model_orbitals, compute_ao_min_orbitals

# def run_casscf_calculation(args):
#   geometry_file, basis, guess_orbitals = args

#   molecule = gto.M(atom=geometry_file,
#                    basis=basis,
#                    spin=0,
#                    symmetry=True)
#   molecule.verbose = 0

#   hartree_fock = molecule.RHF()
#   n_states = 2
#   weights = np.ones(n_states) / n_states
#   casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
#   casscf.conv_tol = 1e-8

#   tic = time.perf_counter()
#   _, _, _, iinner, _, _, _, _ = casscf.kernel(guess_orbitals)
#   toc = time.perf_counter()
#   return iinner, toc - tic

# def construct_casscf_object(geometry_file: str, basis='sto-6g'):
#   molecule = gto.M(atom=geometry_file,
#                    basis=basis,
#                    spin=0,
#                    symmetry=True)
#   molecule.verbose = 0

#   hartree_fock = molecule.RHF()
#   n_states = 2
#   weights = np.ones(n_states) / n_states
#   casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
#   casscf.conv_tol = 1e-8
#   return casscf

# if __name__ == "__main__":
#   base_dir = os.environ['base_dir']
#   geometry_file: str = base_dir + 'geometries/geom_scan_200/geometry_197.xyz'
#   F_model = './checkpoints/' + 'gs200_sto_6g_F' + '.pt'
#   basis = 'sto_6g'

#   _, mo = compute_ao_min_orbitals('', geometry_file, 'sto_6g')
#   for _ in range(4):
#     print(run_casscf_calculation((geometry_file, basis, mo)))

#   print('\n\n')

#   _, mo = compute_F_model_orbitals(F_model, geometry_file, basis)
#   for _ in range(4):
#     print(run_casscf_calculation((geometry_file, basis, mo)))