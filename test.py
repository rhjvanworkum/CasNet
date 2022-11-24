from phisnet_fork.utils.transform_hamiltonians import transform_hamiltonians_from_ao_to_lm, transform_hamiltonians_from_lm_to_ao
from phisnet_fork.utils.definitions import orbital_conventions

import numpy as np

hamiltonian = np.array([
  [0, 2, 3, 4, 5, 6, 7],
  [1, 2, 3, 4, 5, 6, 7],
  [2, 2, 3, 4, 5, 6, 7],
  [3, 2, 3, 4, 5, 6, 7],
  [4, 2, 3, 4, 5, 6, 7],
  [5, 2, 3, 4, 5, 6, 7],
  [6, 2, 3, 4, 5, 6, 7]
])

a = transform_hamiltonians_from_ao_to_lm(hamiltonian, atoms='CHH', convention='fulvene_minimal_basis')
b = transform_hamiltonians_from_lm_to_ao(a, atoms='CHH', convention='fulvene_minimal_basis')
print(b)