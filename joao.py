from ase.db import connect
from pyscf import gto
from pyscf.tools import molden
import numpy as np

def write_db_entry_to_molden_file(molden_file,
                                  atom_numbers,
                                  atom_positions,
                                  mo_coeffs):

    atom_string = ""
    for atom, position in zip(atom_numbers, atom_positions):
      atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

    molecule = gto.M(atom=atom_string,
                    basis='sto_6g',
                    spin=0,
                    symmetry=True)

    with open(molden_file, 'w') as f:
        molden.header(molecule, f)
        molden.orbital_coeff(molecule, f, mo_coeffs, ene=np.zeros(mo_coeffs.shape[0]))

# e_tots = []
# with connect('./data_storage/normal_dist_10k_s0.1_energies.db') as conn:
#   for i in range(9997):
#     e_tots.append(conn.get(i + 1).data['e_tot'])
# np.save('e_tots', np.array(e_tots))


e_tots = np.load('e_tots.npy')

lower = np.mean(e_tots) -  1 * np.std(e_tots)
upper = np.mean(e_tots) +  1 * np.std(e_tots)

outliers = []
for idx, e in enumerate(e_tots):
  if e < lower or e > upper:
    outliers.append(idx)

with connect('normal_dist_10k_s0.1.db') as conn:
  # get the first calculation
  row_1 = conn.get(int(outliers[4] + 1))

  # get mo coeffs and reshape corresponding array
  # columns correspond to mo_1, mo_2, mo_3, etc.
  # rows correspond to basis set coefficients
  mo_coeffs = row_1.data['mo_coeffs'].reshape(36, 36)

  write_db_entry_to_molden_file('test.molden',
                                row_1['numbers'],
                                row_1['positions'],
                                mo_coeffs)





