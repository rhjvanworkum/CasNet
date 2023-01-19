#!/bin/bash

export WANDB_PROJECT='mo_coeff_test'

PYTHONPATH="/home/rhjvanworkum/caschnet/:/home/rhjvanworkum/schnetpack/src/:$PYTHONPATH"
export PYTHONPATH


python train_model.py --db_name fulvene_s005.db --split_name fulvene_normal_5000.npz --property F --model_name testje

# python phisnet_fork/train.py @phisnet_fork/configurations/fulvene_gs250_cc-pVDZ_molcas.txt