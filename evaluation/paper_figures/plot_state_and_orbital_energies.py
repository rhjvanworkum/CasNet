import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable

from data.utils import find_all_geometry_files_in_folder, sort_geometry_files_by_idx
from evaluation import run_casscf_calculation
from evaluation.pyscf import initial_guess_dict

""" Colors """
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_dict = {
    'hartree-fock': colors[1],
    'ML-MO': colors[0],
    'ML-F': colors[2],
    'PhiSNet': colors[3],
}
alpha_dict = {
    'hartree-fock': 0.5,
    'ML-MO': 1.0,
    'ML-F': 0.5,
    'PhiSNet': 0.5,
}

""" Parameters for GS250 and MD25 """
mo_models = [
    f'./checkpoints/geom_scan_250/fulvene_gs250_MO.pt',
    f'./checkpoints/md_traj/fulvene_mdtraj_MO.pt'
]
f_models = [
    f'./checkpoints/geom_scan_250/fulvene_gs250_F.pt',
    f'./checkpoints/md_traj/fulvene_mdtraj_F.pt'
]
phisnet_models =[
    f'./checkpoints/geom_scan_250/fulvene_gs250_phisnet.pt',
    f'./checkpoints/md_traj/fulvene_mdtraj_phisnet.pt'
]

geometry_folders = [
    os.environ['base_dir'] + 'geometries/fulvene_geom_scan_250/',
    os.environ['base_dir'] + 'geometries/fulvene_md_traj_25/'
]
split_names = [
    './data_storage/fulvene_gs_250_inter.npz',
    None
]
basis = 'sto_6g'
use_splits = [
    True,
    False
]

def get_geometry_files(
    folder: str,
    split_name: str,
    use_split: bool
) -> List[str]:
    geometry_files = find_all_geometry_files_in_folder(folder)
    geometry_files = sort_geometry_files_by_idx(geometry_files)

    if not use_split:
        geometry_files = np.array(geometry_files)
    else:
        geometry_files = np.array(geometry_files)[np.load(split_name)['test_idx']]   

    return geometry_files

if __name__ == "__main__":
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
                'figure.figsize': (15, 5),
                'axes.labelsize': 'x-large',
                'axes.titlesize':'x-large',
                'xtick.labelsize':'x-large',
                'ytick.labelsize':'x-large'}
    pylab.rcParams.update(params)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [2, 1]})

    # 1. CASSCF energies
    for nidx in range(2):
        geometry_files = get_geometry_files(geometry_folders[nidx], split_names[nidx], use_splits[nidx])
        x = np.arange(len(geometry_files)).astype(float)
        
        for model, method in zip(
            [None, mo_models[nidx], f_models[nidx], phisnet_models[nidx]],
            ['hartree-fock', 'ML-MO', 'ML-F', 'PhiSNet'],
        ):
            if os.path.exists(f'./results/plots/e_tots_{method}_{nidx}.npy'):
                e_tots = np.load(f'./results/plots/e_tots_{method}_{nidx}.npy')
            else:
                e_tots = []
                for idx, geometry_file in enumerate(geometry_files):
                    _, mo = initial_guess_dict[method](model, geometry_file, basis)
                    conv, e_tot, imacro, imicro, iinner, _, _ = run_casscf_calculation(geometry_file, mo, basis=basis)
                    print(f'{method} at calc {idx}: converged: {conv} {imacro} - {imicro} - {iinner}')
                    e_tots.append(e_tot)
                np.save(f'./results/plots/e_tots_{method}_{nidx}.npy', np.array(e_tots))

            axs[nidx, 0].bar(x, e_tots, width=0.20, label=method, color=color_dict[method], alpha=alpha_dict[method])
            axs[nidx, 0].set_xlabel('Test geometry idx')
            axs[nidx, 0].set_ylabel('CASSCF energy (Hartree)')
            x += 0.20

    axs[0, 0].set_ylim([-240, -220])
    axs[1, 0].set_ylim([-240, -20])

    # 2. Orbital energies
    for nidx in range(2):
        geometry_files = get_geometry_files(geometry_folders[nidx], split_names[nidx], use_splits[nidx])
        
        for model, method in zip(
            [mo_models[nidx], f_models[nidx], phisnet_models[nidx]],
            ['ML-MO', 'ML-F', 'PhiSNet'],
        ):
            if os.path.exists(f'./results/plots/orb_errors_{method}_{nidx}.npy'):
                orb_errors = np.load(f'./results/plots/orb_errors_{method}_{nidx}.npy')
            else:
                orb_errors = []
                for idx, geometry_file in enumerate(geometry_files):
                    pred_energies, pred_orbitals = initial_guess_dict[method](model, geometry_file, basis)
                    conv, _, _, _, _, _, ref_energies = run_casscf_calculation(geometry_file, pred_orbitals, basis)
                    orb_errors.append(np.abs(ref_energies - pred_energies))
                    print(f'{method} at calc {idx}: converged: {conv}')
                np.save(f'./results/plots/orb_errors_{method}_{nidx}.npy', np.array(orb_errors))

            orb_errors = np.mean(np.array(orb_errors), axis=0)
            axs[nidx, 1].plot(np.arange(len(orb_errors)), orb_errors, label=method, color=color_dict[method])
            axs[nidx, 1].set_xlabel('MO idx')
            axs[nidx, 1].set_ylabel('MO energy error (Hartree)')
            axs[nidx, 1].set_yscale("log")
        
    fig.legend(
        labels=['hartree-fock', 'ML-MO', 'ML-F', 'PhiSNet'],
        loc='right', 
    )
    plt.subplots_adjust(
        left=0.1,
        right=0.825,
        top=0.93,
        bottom=0.08,
        wspace=0.30,
        hspace=0.25
    )
    plt.subplot_tool()
    plt.show()