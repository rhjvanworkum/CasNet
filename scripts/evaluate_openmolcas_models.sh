#!/bin/bash
# don't forget to set the --all flag for md trajs

source env.sh

db_name=fulvene_gs_2500_cc-pVDZ_molcas.db
test_geometry_folder=geometries/fulvene_geom_scan_250/
output_folder=openmolcas/test4/
split_name=fulvene_gs_250_inter.npz

# phisnet_model=fulvene_gs250_molcas_phisnet
# basis=ANO-S-MB

phisnet_model=fulvene_gs250_cc-pVDZ_molcas_phisnet_test_4
# phisnet_model=fulvene_gs2500_cc-pVDZ_molcas_phisnet
f_model=fulvene_gs250_cc-pVDZ_openmolcas
basis=cc-pVDZ

# phisnet model loss (train,val,test): 9.733455105198037e-05, 2.7473884806794927e-06, 2.879264924019649e-06

# python evaluation/openmolcas/evaluate_model_loss.py --db_name $db_name --split_name $split_name --phisnet_model $phisnet_model --f_model $f_model
python evaluation/openmolcas/evaluate_orbital_guess_convergence.py --geometry_folder $test_geometry_folder --output_folder $output_folder --split_name $split_name --phisnet_model $phisnet_model --f_model $f_model --basis $basis