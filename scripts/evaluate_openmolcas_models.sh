#!/bin/bash
# don't forget to set the --all flag for md trajs

source env.sh

db_name=fulvene_gs_250_openmolcas.db
test_geometry_folder=geometries/fulvene_geom_scan_250/
output_folder=openmolcas/test1/
split_name=fulvene_gs_250_inter.npz
phisnet_model=fulvene_gs250_molcas_phisnet
basis=ANO-S-MB

python openmolcas/evaluation/evaluate_model_loss.py --db_name $db_name --split_name $split_name --phisnet_model $phisnet_model
python openmolcas/evaluation/evaluate_orbital_guess_convergence.py --geometry_folder $test_geometry_folder --output_folder $output_folder --split_name $split_name --phisnet_model $phisnet_model --basis $basis