#!/bin/bash
# don't forget to set the --all flag for md trajs

source env.sh

test_geometry_folder=geometries/fulvene_geom_scan_250/

db_name=fulvene_geom_scan_250.db
split_name=fulvene_gs_250_inter.npz

# mode=ML-MO
# model=geom_scan_250/fulvene_gs250_MO

mode=hartree-fock
model=geom_scan_250/fulvene_gs250_phisnet
basis=sto_6g

# model=fulvene_s005_200_vdz_phisnet_2
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# model=s005_vdz/fulvene_s005_1000_vdz_phisnet
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# model=s005_vdz/fulvene_s005_5000_vdz_phisnet
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# mode=ao-min
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true

# mode=hartree-fock
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true


# run evaluation scripts
# python evaluation/pyscf/evaluate_model_loss.py --db_name $db_name --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model
python evaluation/pyscf/evaluate_orbital_guesses_convergence.py --geometry_folder $test_geometry_folder --split_name $split_name --model $model --mode $mode --basis $basis
# python evaluation/pyscf/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mode $mode --model $model --basis $basis --all true


# split_name=fulvene_normal_1000.npz
# phisnet_model=fulvene_s01_1000_vdz_phisnet
# python evaluation/pyscf/evaluate_model_loss.py --db_name $db_name --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model
# python evaluation/pyscf/evaluate_orbital_guesses_convergence.py --geometry_folder $test_geometry_folder --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model --basis $basis --all true


# split_name=fulvene_normal_5000.npz
# phisnet_model=fulvene_s01_5000_vdz_phisnet
# python evaluation/pyscf/evaluate_model_loss.py --db_name $db_name --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model
# python evaluation/pyscf/evaluate_orbital_guesses_convergence.py --geometry_folder $test_geometry_folder --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model --basis $basis --all true