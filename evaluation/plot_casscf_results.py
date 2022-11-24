"""
example usage: python evaluation/plot_casscf_results.py --output_folder pyscf/fulvene_geom_scan_250/ --db_name fulvene_geom_scan_250.db
"""

import argparse
import os
from ase.db import connect
import numpy as np
import matplotlib.pyplot as plt

from data.utils import find_all_files_in_output_folder

def plot_matrix_elements(db_path: str, property_name: str):
    matrices = []

    n = 36
    with connect(db_path) as conn:
        for i in range(250):
            F = conn.get(i + 1).data[property_name].reshape(n, n).copy()
            matrices.append(F)
        matrices = np.array(matrices)

    fig, axs = plt.subplots(6, 6)

    for i in range(6):
        for j in range(6):
            axs[i, j].plot(matrices[:, 0, i * 6 + j])
            axs[i, j].axis('off')

    plt.title(f'{property_name} matrix elements over geom scan')

    fig.savefig(f'./results/{property_name}_matrix_elements.png')
    plt.clf()

def plot_averaged_casscf_energy(output_folder: str) -> None:
  figure_name = output_folder.split('/')[-2] + '_average_casscf_energy.png'

  output_folder = base_dir + args.output_folder
  casscf_results = find_all_files_in_output_folder(output_folder)

  # sort the result on index
  casscf_results = list(sorted(casscf_results, key=lambda x: x.index))

  plt.plot(np.arange(len(casscf_results)), [casscf_result.e_tot for casscf_result in casscf_results])
  
  plt.xlabel('Calculation idx')
  plt.ylabel('Energy (Ha)')
  plt.title('Averaged CASSCF energy')
  plt.savefig('./results/' + figure_name)
  plt.clf()



if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--db_name', type=str)
  parser.add_argument('--output_folder', type=str)
  args = parser.parse_args()

  db_path = "./data_storage/" + args.db_name
  plot_matrix_elements(db_path, "F")
  plot_matrix_elements(db_path, "mo_coeffs")
  plot_matrix_elements(db_path, "mo_coeffs_adjusted")
  output_folder = base_dir + args.output_folder
  plot_averaged_casscf_energy(output_folder)