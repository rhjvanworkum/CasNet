#!/bin/bash

export OMP_NUM_THREADS=8

python data/run_casscf_calculations.py --geometry_folder geometries/fulvene_s01/ --output_folder pyscf/fulvene_s01_vdz/ --basis cc-pVDZ