#!/bin/bash

# load dirs
source env.sh

# env variables

# variables
test_geometry_folder=geometries/fulvene_geom_scan_250/
db_name=fulvene_geom_scan_250.db
split_name=fulvene_gs_250_inter.npz
mo_model=fulvene_gs250_inter_MO
f_model=fulvene_gs250_inter_F
phisnet_model=fulvene_gs250_inter_phisnet
basis=sto_6g

python evaluation/evaluate_model_loss.py --db_name $db_name --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model
python evaluation/evaluate_orbital_guesses_convergence.py --geometry_folder $test_geometry_folder --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model --basis $basis
python evaluation/evaluate_orbital_guesses_energies.py --geometry_folder $test_geometry_folder --split_name $split_name --mo_model $mo_model --F_model $f_model --phisnet_model $phisnet_model --basis $basis