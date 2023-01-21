import os
import argparse
import numpy as np
from ase.db import connect
from ase.io import write
import matplotlib.pyplot as plt

from evaluation.pyscf import compute_ao_min_orbitals, compute_casscf_orbitals, compute_mo_model_orbitals, compute_F_model_orbitals, compute_phisnet_model_orbitals



if __name__ == "__main__":
    base_dir = os.environ['base_dir']

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str)
    parser.add_argument('--split_name', type=str)
    parser.add_argument('--mo_model', type=str)
    parser.add_argument('--F_model', type=str)
    parser.add_argument('--phisnet_model', type=str)
    parser.add_argument('--basis', type=str)
    args = parser.parse_args()

    db_name = './data_storage/' + args.db_name
    split_file = './data_storage/' + args.split_name
    mo_model = './checkpoints/' + args.mo_model + '.pt'
    f_model = './checkpoints/' + args.F_model + '.pt'
    phisnet_model = './checkpoints/' + args.phisnet_model + '.pt'
    basis = args.basis

    # ao_min_errors = []
    mo_errors = []
    f_errors = []
    phisnet_errors = []

    split = np.load(split_file)
    test_idx = split['test_idx']

    for idx in test_idx:
        with connect(db_name) as conn:
            atoms = conn.get_atoms(idx=int(idx))
            write('temp.xyz', atoms)

            ref_energies, _ = compute_casscf_orbitals(None, 'temp.xyz', basis)

            # ao_min_energies, _ = compute_ao_min_orbitals(None, 'temp.xyz', basis, compute_cas_orb_e=False)
            # ao_min_errors.append(np.abs(ref_energies - ao_min_energies))

            mo_energies, _ = compute_mo_model_orbitals(mo_model, 'temp.xyz', basis, compute_cas_orb_e=True)
            mo_errors.append(np.abs(ref_energies - mo_energies))

            f_energies, _ = compute_F_model_orbitals(f_model, 'temp.xyz', basis)
            f_errors.append(np.abs(ref_energies - f_energies))

            phisnet_energies, _ = compute_phisnet_model_orbitals(phisnet_model, 'temp.xyz', basis)
            phisnet_errors.append(np.abs(ref_energies - phisnet_energies))

    
    for label, errors in zip(
        ['mo', 'F', 'phisnet'],
        [mo_errors, f_errors, phisnet_errors]   
    ):
        errors = np.mean(errors, axis=0)
        plt.plot(np.arange(len(errors)), errors, label=label)

    plt.yscale("log")
    plt.legend()
    plt.show()

            