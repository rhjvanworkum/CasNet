import os 
import shutil


for idx, i in enumerate(range(39, 1000, 40)):
  # print(idx, i)
  shutil.copy2(src=f'/home/ruard/Documents/experiments/fulvene/geometries/MD_trajectories_5_01/geometry_{i}.xyz',
               dst=f'/home/ruard/Documents/experiments/fulvene/geometries/fulvene_md_traj_25/geometry_{idx}.xyz')








# from itertools import permutations, combinations
# from typing import Any
# from ase.db import connect
# from base64 import b64decode
# from data.utils import find_all_files_in_output_folder
# import numpy as np
# from pyscf import gto
# import matplotlib.pyplot as plt
# from pyscf.tools import molden
# import scipy
# from tqdm import tqdm
# import scipy.linalg

# def write_db_entry_to_molden_file(molden_file_name: str, db_row: Any, mo_coeffs):
#     atom_string = ""
#     for atom, position in zip(db_row["numbers"], db_row["positions"]):
#       atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

#     molecule = gto.M(atom=atom_string,
#                     basis='sto_6g',
#                     spin=0,
#                     symmetry=False)

#     with open(molden_file_name, 'w') as f:
#         molden.header(molecule, f)
#         molden.orbital_coeff(molecule, f, mo_coeffs, ene=np.zeros(mo_coeffs.shape[0]))

# """ Write tot energies to plot """
# # casscf_results = find_all_files_in_output_folder('/home/ruard/Documents/experiments/fulvene/pyscf/ethene_geom_scan/')
# # casscf_results = list(sorted(casscf_results, key=lambda x: x.index))
# # e = [result.e_tot for result in casscf_results]
# # # plt.hist(e, bins=50)
# # plt.plot(e)
# # plt.show()

# """ Write orbital files """
# # with connect('./data_storage/ethene_geom_scan.db') as conn:
# #   for i in range(200):
# #     row = conn.get(i+1)
# #     mo_coeffs = row.data['mo_coeffs'].reshape(14, 14)
# #     write_db_entry_to_molden_file(f'./orbitals/{i}_file.molden', row, mo_coeffs)

# """ Plot F matrix entries """
# F_matrices = []

# n = 36
# with connect('./data_storage/test.db') as conn:
#   for i in range(250):
#     indices = []
#     F = conn.get(i + 1).data['mo_coeffs_adjusted'].reshape(n, n).copy()
#     F_matrices.append(F)
# F_matrices = np.array(F_matrices)

# for i in range(n):
#   for j in range(n):
#     plt.plot(F_matrices[:, i, j])
#     plt.savefig(f'./results/matrices/fig_{i}_{j}.png')
#     plt.clf()





# # prev_F = None
# # threshold = 5
# # prev_change = None
# # with connect('./data_storage/ethene_test.db') as conn:
# #   for i in range(200):
# #     indices = []
# #     F = conn.get(i + 1).data['F'].reshape(14, 14).copy()
# #     if prev_F is not None:
# #       diff = np.abs(prev_F - F) / np.abs(prev_F) * 100
# #       for idx in range(14):
# #         for jdx in range(14):
# #           if diff[idx, jdx] > threshold:
# #             indices.append((idx, jdx))

#     # for ind in indices:
#     #   if (ind[1], ind[0]) in indices:
#     #     indices.remove((ind[1], ind[0]))

#     # five_list = [(i, 9) for i in range(14)]
#     # print(f'index {i}')
#     # for ind in indices:
#     #   if ind in five_list:
#     #     print(ind)
#     # # print('\n\n')

#     # new_F = F.copy()

#     # """Between 130-132 & 155-199 AO 5 & 6 switch"""
#     # if i >= 130 and i < 132:
#     #   for i in range(14):
#     #     new_F[i, 5] = F[i, 6]
#     #     new_F[i, 6] = F[i, 5]
#     # if i >= 155 and i < 199:
#     #   for i in range(14):
#     #     new_F[i, 5] = F[i, 6]
#     #     new_F[i, 6] = F[i, 5]

#     # """ Between 89-132 switch AO 9 & 10 """
#     # if i >= 89 and i < 133:
#     #   for i in range(14):
#     #     new_F[i, 9] = F[i, 10]
#     #     new_F[i, 10] = F[i, 9]

#     # """ Between 155-199 do AO 9 -> AO 11, AO 11 -> AO 10, AO 10 -> AO 9 """
#     # if i >= 155 and i < 199:
#     #   for i in range(14):
#     #     new_F[i, 9] = F[i, 10]
#     #     new_F[i, 10] = F[i, 11]
#     #     new_F[i, 11] = F[i, 9]

#     # """Between 26-83,89-132,155-199 &  AO 12 & 13 switch"""
#     # if i >= 26 and i < 83:
#     #   for i in range(14):
#     #     new_F[i, 12] = F[i, 13]
#     #     new_F[i, 13] = F[i, 12]
#     # if i >= 89 and i < 132:
#     #   for i in range(14):
#     #     new_F[i, 12] = F[i, 13]
#     #     new_F[i, 13] = F[i, 12]
#     # if i >= 155 and i < 199:
#     #   for i in range(14):
#     #     new_F[i, 12] = F[i, 13]
#     #     new_F[i, 13] = F[i, 12]

#     # # if (0, 5) in indices:
#     # print(i, len(indices))

#     # if len(indices) > 0 and len(indices) < 11:
#     #   for comb in permutations(indices, len(indices)):
#     #     prev_Fs = [prev_F[com[0], com[1]] for com in comb]
        
#     #     new_Fs = []
#     #     for i in range(len(comb) - 1):
#     #       new_Fs.append(F[comb[i + 1][0], comb[i + 1][1]].copy())
#     #     new_Fs.append(F[comb[0][0], comb[0][1]].copy())

#     #     if False not in [abs(new_Fs[i] - prev_Fs[i]) / abs(new_Fs[i]) * 100 > threshold for i in range(len(comb))]:
#     #       print('Found match')
          
#     #       for com, F_el in zip(comb, new_Fs):
#     #         F[com[0], com[1]] = F_el
#     #       break
#     # elif len(indices) > 0 and len(indices) <= 11:
#     #   for ind in indices:
#     #     F[ind[0], ind[1]] = prev_F[ind[0], ind[1]]

#     # print(F[0, 5])

#     # F_matrices.append(new_F.tolist())
#     # prev_F = F
# # for i in range(200):
# #   F_matrices.append(np.load(f'/mnt/c/users/rhjva/imperial/fulvene/pyscf/ethene_test_2/geometry_{i}.npz')['F'][0])

# # # with connect('h2o_hamiltonians.db') as conn:
# # #   for i in range(200):
# # #     row = conn.get(i + 1)
# # #     shape = row.data['_shape_hamiltonian']
# # #     dtype = row.data['_dtype_hamiltonian']
# # #     ham = np.frombuffer(b64decode(row.data['hamiltonian']), dtype=dtype)
# # #     ham = ham.reshape(shape)
# # #     F_matrices.append(ham.tolist())
# # # with connect('./data_storage/normal_dist_10k_s0.1.db') as conn:
# # #   for i in range(200):
# # #     index = int(np.random.choice(np.arange(9000)))
# # #     row = conn.get(index + 1)
# # #     F_matrices.append(row.data['F'].reshape(36, 36).tolist())

# # F_matrices = np.array(F_matrices)
# # for i in range(14):
# #   for j in range(14):
# #     plt.plot(F_matrices[:, i, j])
# #     plt.savefig(f'./f_matrices_2/fig_{i}_{j}.png')
# #     plt.clf()

# """ Plot mo energies """
# # energies = []
# # with connect('./data_storage/ethene_test.db') as conn:
# #   for i in range(200):
# #     energies.append(conn.get(i+1).data['mo_energies'].tolist())
# # energies = np.array(energies)
# # for i in range(14):
# #   plt.plot(energies[:, i])
  
# # plt.savefig(f'./mo_energies/tot.png')
# # plt.clf()

# """ Orbital order """
# # F_matrices = []
# # with connect('./data_storage/ethene_test.db') as conn:
# #   initial_orbitals = conn.get(1).data['mo_coeffs'].reshape(14, 14)
# #   for i in range(200):
# #     new_orbitals = conn.get(i+1).data['mo_coeffs'].reshape(14, 14)

# #     orb_order = get_orbital_order(initial_orbitals.T, new_orbitals.T)
# #     if len(set(orb_order)) != 14:
# #       missing_idx = np.argwhere(np.array([n not in orb_order for n in range(14)]))[0][0]
# #       counts = [orb_order.tolist().count(n) for n in range(14)]
# #       replace_number = np.argwhere(np.array(counts) == 2)[0][0]
# #       replace_idxs = np.argwhere(orb_order == replace_number)
# #       if replace_number < missing_idx:
# #         orb_order[replace_idxs[1][0]] = missing_idx
# #       else:
# #         orb_order[replace_idxs[0][0]] = missing_idx
    
# #     for num in range(14):
# #       assert num in orb_order

# #     F_matrix = conn.get(i+1).data['F'].reshape(14, 14).copy()
# #     F_matrix = sort_fock_matrix(F_matrix, orb_order=orb_order)
# #     F_matrices.append(F_matrix)

# # F_matrices = np.array(F_matrices)
# # for i in range(14):
# #   for j in range(14):
# #     plt.plot(F_matrices[:, i, j])
# #     plt.savefig(f'./f_matrices/fig_{i}_{j}.png')
# #     plt.clf()


# # plt.plot(np.arange(len(e)), e)
# # plt.savefig('test.png')


# # def perform_casscf_calc(db_row, mo_coeffs):
# #     atom_string = ""
# #     for atom, position in zip(db_row["numbers"], db_row["positions"]):
# #       atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

# #     molecule = gto.M(atom=atom_string,
# #                     basis='sto_6g',
# #                     spin=0,
# #                     symmetry=True)

# #     hartree_fock = molecule.RHF()
# #     n_states = 2
# #     weights = np.ones(n_states) / n_states
# #     casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
# #     casscf.conv_tol = 1e-8

# #     mo_coeffs = casscf.sort_mo([19, 20, 21, 22, 23, 24], mo_coeffs)
# #     e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mo_coeffs)
# #     print(f'e_tot: {e_tot}')
# #     return imacro, imicro, iinner

# # if __name__ == "__main__":
# #   calc_idx = 180

# #   with connect('data_storage/geom_scan_200_sto_6g.db') as conn:
# #     # get reference calc
# #     row = conn.get(1)
# #     ref_mo_coeffs = row.data["mo_coeffs"].reshape(36, 36)
# #     write_db_entry_to_molden_file("ref_calc.molden", row, ref_mo_coeffs)

# #     # get other calc
# #     row = conn.get(calc_idx + 1)
# #     mo_coeffs = row.data["mo_coeffs"].reshape(36, 36)
# #     # adjusted_mo_coeffs = row.data["mo_coeffs_adjusted"].reshape(36, 36)
# #     # orb_order = get_orbital_order(ref_mo_coeffs.T, mo_coeffs.T)
# #     # print(orb_order + 1)
# #     write_db_entry_to_molden_file("calc_initial.molden", row, mo_coeffs)
# #     # result = perform_casscf_calc(row, mo_coeffs)
# #     # print(result)
# #     # print('\n')


# #     orb_order = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 16,
# #                  18, 20, 19, 21, 23, 22, 25, 24, 28, 29, 26, 27, 30, 31, 32, 33, 34, 35]
# #     # adjusted_mo_coeffs = mo_coeffs.T[orb_order].T
# #     F = row.data["F"].reshape(36, 36).copy()
# #     S = row.data["S"].reshape(36, 36)

# #     F_adjusted = sort_fock_matrix(orb_order, F)
# #     S_adjusted = sort_fock_matrix(orb_order, S)

# #     _, mo_coeffs = scipy.linalg.eigh(F_adjusted, S_adjusted)


# #     write_db_entry_to_molden_file("calc_adjusted.molden", row, mo_coeffs)
# #     # result = perform_casscf_calc(row, adjusted_mo_coeffs)
# #     # print(result)