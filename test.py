
import multiprocessing
import os
from typing import List, Optional
import argparse
import numpy as np
from pyscf import gto, mcscf
from tqdm import tqdm
from functools import reduce

from data.utils import CasscfResult, check_and_create_folder, find_all_geometry_files_in_folder, sort_geometry_files_by_idx, sort_geometry_files_by_distance

guess_mos = None

molecule = gto.M(atom='/home/rhjvanworkum/geometry_0.xyz',
                   basis='cc-pVDZ',
                   spin=0,
                   symmetry=True)

hartree_fock = molecule.RHF()
hartree_fock.kernel()
S = hartree_fock.get_ovlp(molecule)

print(hartree_fock.mo_energy, hartree_fock.mo_coeff)
# print(S.shape)

# n_states = 3
# weights = np.ones(n_states) / n_states
# casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)

# if not guess_mos is None:
#   mo = mcscf.project_init_guess(casscf, guess_mos)
# else: 
#   mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
# mo = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo)

# conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo)

# F = casscf.get_fock()