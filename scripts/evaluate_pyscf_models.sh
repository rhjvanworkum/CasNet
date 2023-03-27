#!/bin/bash
# don't forget to set the --all flag for md trajs

source env.sh

test_geometry_folder=geometries/fulvene_md_traj_25/

db_name=fulvene_geom_scan_250.db
split_name=fulvene_gs_250_inter.npz


mode=ML-MO
model=md_traj/fulvene_mdtraj_MO
basis=sto_6g

python evaluation/pyscf/compute_CASCI_energy_errors.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# model=s005_vdz/fulvene_s005_1000_vdz_phisnet
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# model=s005_vdz/fulvene_s005_5000_vdz_phisnet
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# mode=ao-min
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# mode=hartree-fock
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true