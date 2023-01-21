#!/bin/bash

source env.sh

# GEOM_SCAN_250 plot
db_name=fulvene_geom_scan_250.db
split_name=fulvene_gs_250_inter.npz
mo_model=geom_scan_250/fulvene_gs250_MO
F_model=geom_scan_250/fulvene_gs250_F
phisnet_model=geom_scan_250/fulvene_gs250_phisnet
basis=sto_6g

python evaluation/pyscf/plot_orbital_energies.py --db_name $db_name --split_name $split_name --mo_model $mo_model --F_model $F_model --phisnet_model $phisnet_model --basis $basis


# MD_TRAJ plot
db_name=fulvene_md_250.db
split_name=fulvene_md_250.npz

python evaluation/pyscf/plot_orbital_energies.py --db_name $db_name --split_name $split_name --mo_model $mo_model --F_model $F_model --phisnet_model $phisnet_model --basis $basis