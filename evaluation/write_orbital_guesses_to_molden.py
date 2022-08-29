from evaluation.utils import compute_F_model_orbitals, compute_ao_min_orbitals, compute_converged_casscf_orbitals, compute_huckel_orbitals, compute_mo_model_orbitals
from pyscf import gto
from pyscf.tools import molden
import numpy as np
import argparse
import os

initial_guess_dict = {
  'ao_min': compute_ao_min_orbitals,
  'huckel': compute_huckel_orbitals,
  'ML-MO': compute_mo_model_orbitals,
  'ML-F': compute_F_model_orbitals,
  'converged': compute_converged_casscf_orbitals
}

def write_orbitals_to_molden(molden_file: str, 
                             geometry_file: str,
                             basis: str,
                             orbitals: np.ndarray, 
                             energies: np.ndarray) -> None:

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
  parser.add_argument('--mo_model', type=str)
  parser.add_argument('--F_model', type=str)
  parser.add_argument('--basis', type=str)
  args = parser.parse_args()

  geometry_path = base_dir + args.geometry_path
  mo_model = './checkpoints/' + args.mo_model + '.pt'
  F_model = './checkpoints/' + args.F_model + '.pt'
  basis = args.basis

  for name, method in initial_guess_dict.items():
    model_path = ''
    if name == 'ML-MO':
      model_path = mo_model
    elif name == 'ML-F':
      model_path = F_model
    mo_e, mo = method(model_path, geometry_path, basis)
    write_orbitals_to_molden('results/' + geometry_path.split('/')[-1].split('.')[0] + '_' + name + '.molden', 
                            geometry_path,
                            basis,
                            mo,
                            mo_e)

