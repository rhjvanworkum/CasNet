#!/bin/bash

basis=sto_6g

# # ML-MO
# geometry_path=geometries/fulvene_geom_scan_250/geometry_230.xyz
# name=orbital_mo
# mode=ML-MO
# model=geom_scan_250/fulvene_gs250_MO
# python evaluation/pyscf/write_orbital_guesses_to_molden.py --geometry_path $geometry_path --mode $mode --model $model --basis $basis --name $name

# # ML-F
# geometry_path=geometries/fulvene_geom_scan_250/geometry_230.xyz
# name=orbital_f
# mode=ML-F
# model=geom_scan_250/fulvene_gs250_F
# python evaluation/pyscf/write_orbital_guesses_to_molden.py --geometry_path $geometry_path --mode $mode --model $model --basis $basis --name $name

# # PhisNet
# geometry_path=geometries/fulvene_geom_scan_250/geometry_230.xyz
# name=orbital_phisnet
# mode=phisnet
# model=geom_scan_250/fulvene_gs250_phisnet
# python evaluation/pyscf/write_orbital_guesses_to_molden.py --geometry_path $geometry_path --mode $mode --model $model --basis $basis --name $name

# CASSCF
geometry_path=geometries/fulvene_geom_scan_250/geometry_230.xyz
name=orbital_casscf
mode=casscf
model=none
python evaluation/pyscf/write_orbital_guesses_to_molden.py --geometry_path $geometry_path --mode $mode --model $model --basis $basis --name $name