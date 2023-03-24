"""
Script to plot smoothness of Fock / MO matrix elements


example usage: python evaluation/plot_casscf_results.py --output_folder pyscf/fulvene_geom_scan_250/ --db_name fulvene_geom_scan_250.db
"""

import argparse
import os
from ase.db import connect
import numpy as np
import matplotlib.pyplot as plt

from data.utils import find_all_files_in_output_folder
from data.db.save_casscf_calculations_to_db import phase_correct_orbitals


def plot_matrix_elements(db_path: str, property_name: str):
    matrices = []

    n = 36
    with connect(db_path) as conn:
        for i in range(250):
            prop = conn.get(i + 1).data[property_name].reshape(n, n).copy()
            matrices.append(prop)
        matrices = np.array(matrices)

    fig, axs = plt.subplots(6, 6)

    for i in range(6):
        for j in range(6):
            axs[i, j].plot(matrices[:, 0, i * 6 + j])
            axs[i, j].axis('off')
    
    # add a global x-axis to the bottom of the figure
    global_x = fig.add_subplot(212, frameon=False)
    global_x.annotate(
      '', 
      xy=(1, 0), 
      xytext=(0.0, 0), 
      xycoords='axes fraction', 
      textcoords='axes fraction', 
      arrowprops=dict(arrowstyle='->')
    )
    global_x.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    global_x.set_xlabel('Geometry scan index')

    global_y = fig.add_subplot(221, frameon=False)
    global_y.annotate(
      '', 
      xy=(0, 1.1), 
      xytext=(0, -1.2),
      xycoords='axes fraction', 
      textcoords='axes fraction', 
      arrowprops=dict(arrowstyle='->')
    )
    global_y.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    global_y.set_ylabel('Coefficient')

    plt.show()

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
  db_path = "./data_storage/fulvene_geom_scan_250.db"
  plot_matrix_elements(db_path, "F")
  plot_matrix_elements(db_path, "mo_coeffs")
  plot_matrix_elements(db_path, "mo_coeffs_adjusted")

#   plot_averaged_casscf_energy(output_folder)