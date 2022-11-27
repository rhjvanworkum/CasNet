import os
import argparse
import torch
import numpy as np
from ase.db import connect
from phisnet_fork.utils.transform_hamiltonians import transform_hamiltonians_from_lm_to_ao


def evaluate_phisnet_model_loss(model_path, db_path, split_path):
    
    def calculate_loss(model, db_path, indices):
        loss = 0
        for idx in indices:
            with connect(db_path) as conn:
                target = conn.get(int(idx) + 1).data['F']
                atoms = conn.get_atoms(idx=int(idx))
                R = torch.stack([torch.tensor(atoms.positions * 1.8897261258369282, dtype=torch.float32)]).to(device)
                atom_symbols = ''
                for atom in atoms:
                    atom_symbols += atom.symbol

            pred = model(R=R)['full_hamiltonian'][0].detach().cpu().numpy()
            orbital_convention = 'fulvene_minimal_basis'
            pred = transform_hamiltonians_from_lm_to_ao(pred, atoms=atom_symbols, convention=orbital_convention)
            pred = pred.reshape(-1)
            loss += np.sum((pred - target)**2)
        loss /= len(train_idx) * pred.shape[0]
        return loss
        
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = torch.load(model_path, map_location=device).to(device)
    
    split = np.load(split_path)
    train_idx, val_idx, test_idx = split['train_idx'], split['val_idx'], split['test_idx']
    
    model.train()
    train_loss = calculate_loss(model, db_path, train_idx)

    model.eval()
    val_loss = calculate_loss(model, db_path, val_idx)
    test_loss = calculate_loss(model, db_path, test_idx)
    test_loss = None

    return train_loss, val_loss, test_loss


if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--db_name', type=str)
  parser.add_argument('--split_name', type=str)
  parser.add_argument('--phisnet_model', type=str)
  args = parser.parse_args()

  db_name = './data_storage/' + args.db_name
  split_file = './data_storage/' + args.split_name
  phisnet_model = './checkpoints/' + args.phisnet_model + '.pt'
  
  train_loss, val_loss, test_loss = evaluate_phisnet_model_loss(phisnet_model, db_name, split_file)
  print(f'phisnet model loss (train,val,test): {train_loss}, {val_loss}, {test_loss}')