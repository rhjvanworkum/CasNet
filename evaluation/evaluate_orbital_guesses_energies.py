from evaluation.utils import compute_F_model_orbitals, compute_ao_min_orbitals, compute_casci_energy, compute_converged_casci_energy, compute_converged_casscf_orbitals, compute_huckel_orbitals, compute_mo_model_orbitals
from pyscf import gto
from pyscf.tools import molden
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

initial_guess_dict = {
  'ao_min': compute_ao_min_orbitals,
  'huckel': compute_huckel_orbitals,
  'ML-MO': compute_mo_model_orbitals,
  'ML-F': compute_F_model_orbitals,
}

def plot_mo_energies(geometry_path: str, basis: str, mo_model: str, F_model: str) -> None:
  mo_e_converged, _ = compute_converged_casscf_orbitals(model_path='', 
                                                        geometry_path=geometry_path,
                                                        basis=basis)

  for name, method in initial_guess_dict.items():
    model_path = ''
    if name == 'ML-MO':
      model_path = mo_model
    elif name == 'ML-F':
      model_path = F_model
    mo_e, mo = method(model_path, geometry_path, basis)

    plt.plot(np.arange(len(mo_e)), np.abs(mo_e_converged - mo_e), label=name)
  plt.savefig('results/mo_e.png')

def plot_casci_energies(geometry_path: str, basis: str, mo_model: str, F_model: str) -> None:
  e_conv = compute_converged_casci_energy(geometry_path, basis)

  diff = []
  for name, method in initial_guess_dict.items():
    model_path = ''
    if name == 'ML-MO':
      model_path = mo_model
    elif name == 'ML-F':
      model_path = F_model
    _, mo = method(model_path, geometry_path, basis)
    e_cas = compute_casci_energy(geometry_path, mo , basis)
    diff.append(np.abs(e_conv - e_cas))

  for idx, name in enumerate(list(initial_guess_dict.keys())):
    print(name, diff[idx])

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

  plot_casci_energies(geometry_path, basis, mo_model, F_model)
