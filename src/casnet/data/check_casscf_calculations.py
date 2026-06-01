"""
Plot averaged state energies + MO energies of the geom scan 200
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from casnet.config import get_base_dir
from casnet.data.utils import CasscfResult, find_all_files_in_output_folder


def plot_mo_energies(output_folder: str, casscf_results: list[CasscfResult]) -> None:
    figure_name = output_folder.split("/")[-2] + "_mo_energies.png"

    for casscf_result in casscf_results:
        plt.scatter(
            np.arange(len(casscf_result.mo_energies)), casscf_result.mo_energies, color="blue"
        )

    plt.xlabel("MO idx")
    plt.ylabel("MO energy (Ha)")
    plt.title("MO energies")
    plt.savefig("./results/" + figure_name)
    plt.clf()


def plot_averaged_casscf_energy(output_folder: str, casscf_results: list[CasscfResult]) -> None:
    figure_name = output_folder.split("/")[-2] + "_average_casscf_energy.png"

    # sort the result on index
    casscf_results = list(sorted(casscf_results, key=lambda x: x.index))

    plt.scatter(
        np.arange(len(casscf_results)), [casscf_result.e_tot for casscf_result in casscf_results]
    )

    plt.xlabel("Calculation idx")
    plt.ylabel("Energy (Ha)")
    plt.title("Averaged CASSCF energy")
    plt.savefig("./results/" + figure_name)
    plt.clf()


if __name__ == "__main__":
    base_dir = get_base_dir()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()

    output_folder = base_dir + args.output_folder
    casscf_results = find_all_files_in_output_folder(output_folder)

    plot_mo_energies(output_folder, casscf_results)
    plot_averaged_casscf_energy(output_folder, casscf_results)
