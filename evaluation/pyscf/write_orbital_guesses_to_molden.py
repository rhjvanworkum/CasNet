from evaluation.pyscf import initial_guess_dict
from pyscf import gto
from pyscf.tools import molden
import numpy as np
import argparse
import os

def write_orbitals_to_molden(
  molden_file: str, 
  geometry_file: str,
  basis: str,
  orbitals: np.ndarray, 
  energies: np.ndarray
) -> None:

  molecule = gto.M(atom=geometry_file,
                   basis=basis,
                   spin=0,
                   symmetry=True)

  with open(molden_file, 'w') as f:
      molden.header(molecule, f)
      molden.orbital_coeff(molecule, f, orbitals, ene=energies)


if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_path', type=str)
  parser.add_argument('--mode', type=str)
  parser.add_argument('--model', type=str)
  parser.add_argument('--basis', type=str)
  args = parser.parse_args()

  geometry_path = base_dir + args.geometry_path
  mode = args.mode
  model_path = './checkpoints/' + args.model + '.pt'
  basis = args.basis

  method = initial_guess_dict[mode]
  mo_e, mo = method(model_path, geometry_path, basis)
  write_orbitals_to_molden(
    'results/' + geometry_path.split('/')[-1].split('.')[0] + '_' + '.molden', 
    geometry_path,
    basis,
    mo,
    mo_e
)