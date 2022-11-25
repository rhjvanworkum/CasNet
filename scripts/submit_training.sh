#!/bin/bash

export WANDB_PROJECT='caschnet'

PYTHONPATH="/home/rhjvanworkum/caschnet/:/home/rhjvanworkum/schnetpack/src/:$PYTHONPATH"
export PYTHONPATH


# python train_model.py --db_name fulvene_md_250.db --split_name fulvene_md_250.npz --property F --model_name fulvene_md_250_F_bigger

python phisnet_fork/train.py @phisnet_fork/configurations/fulvene_md250.txt