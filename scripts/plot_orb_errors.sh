#!/bin/bash

source env.sh

# GEOM_SCAN_250 plot
mo_model=geom_scan_250/fulvene_gs250_MO
F_model=geom_scan_250/fulvene_gs250_F
phisnet_model=geom_scan_250/fulvene_gs250_phisnet
basis=sto_6g

python evaluation/pyscf/plot_orbital_energies.py --db_names fulvene_geom_scan_250.db fulvene_md_250.db --split_names fulvene_gs_250_inter.npz fulvene_md_250.npz --mo_model $mo_model --F_model $F_model --phisnet_model $phisnet_model --basis $basis