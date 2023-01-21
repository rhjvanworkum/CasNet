#!/bin/bash

export WANDB_PROJECT='caschnet_ML-F'

PYTHONPATH="/home/rhjvanworkum/caschnet/:/home/rhjvanworkum/schnetpack/src/:$PYTHONPATH"
export PYTHONPATH


python train_model.py --db_name fulvene_s005.db --split_name fulvene_normal_200.npz --property F --model_name fulvene_s005_200_ML-F_test

# python phisnet_fork/train.py @phisnet_fork/configurations/fulvene_gs250_cc-pVDZ_molcas.txt