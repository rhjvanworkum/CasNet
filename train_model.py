"""
Script for training NN on CAS orbitals
"""
import argparse
from model.caschnet_model import create_orbital_model

from model.loss_functions import mean_squared_error, symm_matrix_mse
from model.training import train_model


if __name__ == "__main__":
  epochs = 100
  lr = 1e-3
  batch_size = 16
  cutoff = 5.0
  basis_set_size = 36
  use_wandb = True

  parser = argparse.ArgumentParser()
  parser.add_argument('--db_name', type=str)
  parser.add_argument('--split_name', type=str)
  parser.add_argument('--property', type=str)
  parser.add_argument('--model_name', type=str)
  args = parser.parse_args()

  database_path = './data_storage/' + args.db_name
  split_file = './data_storage/' + args.split_name
  model_name = args.model_name
  create_model_fn = create_orbital_model

  property = args.property
  if property == 'mo_coeffs' or property == 'mo_coeffs_adjusted' or property == 'dm':
    loss_fn = mean_squared_error
  elif property == 'F':
    loss_fn = symm_matrix_mse

  train_model(save_path='./checkpoints/' + model_name + '.pt',
                  property=property, 
                  loss_fn=loss_fn, 
                  batch_size=batch_size, 
                  lr=lr, 
                  epochs=epochs,
                  basis_set_size=basis_set_size,
                  database_path=database_path,
                  create_model_fn=create_model_fn,
                  split_file=split_file,
                  use_wandb=use_wandb,
                  cutoff=cutoff) 