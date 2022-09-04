"""
Temp script to evaluate SchNorb predictions
2 modes:
- without_S -> uses S from PySCF
- with_S -> uses S from SchNorb prediction
"""
import numpy as np
import scipy
import scipy.linalg
from evaluation.evaluate_orbital_guesses_convergence import run_casscf_calculation
from evaluation.utils import compute_casci_energy, compute_converged_casci_energy, compute_converged_casscf_orbitals
import matplotlib.pyplot as plt

from model.inference import calculate_overlap_matrix

def infer_orbitals(predictions, idx, geometry_path, with_S=False):
  F = predictions['hamiltonian'][idx]
  if with_S:
    S = predictions['overlap'][idx].astype('float64')

  else:
    S = calculate_overlap_matrix(geometry_path, basis='sto-6g')
  
  # F -> MO coeffs
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo_e, mo

if __name__ == "__main__":
  path = '/mnt/c/users/rhjva/imperial/SchNOrb/quambo_model_gs200/prediction.npz'
  predictions = np.load(path)
  basis = 'sto-6g'
  geometries = np.load('./data_storage/geom_scan_200.npz')['val_idx']
  with_S = False

  """ Perform Evaluations """
  macro_iterations = []
  micro_iterations = []
  inner_iterations = []
  mo_e_errors = []
  e_cas_errors = []
  for idx, geometry_idx in enumerate(geometries):
    geometry_path = f'/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/geometry_{geometry_idx}.xyz'
    mo_e, mo = infer_orbitals(predictions, idx, geometry_path, with_S=with_S)
    
    # iterations
    imacro, imicro, iinner = run_casscf_calculation(geometry_path, mo)
    print(f'SchNorb at calc {idx}: {imacro} - {imicro} - {iinner}')
    macro_iterations.append(imacro)
    micro_iterations.append(imicro)
    inner_iterations.append(iinner)

    # MO energy errors
    mo_e_converged, _ = compute_converged_casscf_orbitals(model_path='', 
                                                          geometry_path=geometry_path,
                                                          basis=basis)
    mo_e_errors.append(np.abs(mo_e_converged - mo_e))

    # CASCI energy errrors
    e_conv = compute_converged_casci_energy(geometry_path, basis)
    e_cas = compute_casci_energy(geometry_path, mo , basis)
    e_cas_errors.append(np.abs(e_conv - e_cas))


  """ Save Results """
  # print out convergence statistics
  print(f'Method SchNorb convergence: \n \
        Macro iterations: {np.mean(np.array(macro_iterations))} +/- {np.std(np.array(macro_iterations))} \n \
        Micro iterations: {np.mean(np.array(micro_iterations))} +/- {np.std(np.array(micro_iterations))} \n \
        Inner iterations: {np.mean(np.array(inner_iterations))} +/- {np.std(np.array(inner_iterations))} \n')
  # Casci energies
  print(f'Method SchNOrb CASCI MAE: {np.mean(np.array(e_cas_errors))} +/- {np.std(np.array(e_cas_errors))} \n')
  # plot mo e energies
  plt.plot(np.arange(len(mo_e_errors)), mo_e_errors, label='SchNorb')
  plt.savefig(f'results/SchNOrb_mo_e.png')
