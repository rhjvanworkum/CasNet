import numpy as np
import torch
import schnetpack as spk
from ase import io

from data.utils import read_xyz_file
from phisnet_fork.utils.transform_hamiltonians import transform_hamiltonians_from_lm_to_ao

def infer_orbitals_from_phisnet_model(model_path: str, 
                                      geometry_path: str,
                                      orbital_convention: str = 'fulvene_minimal_basis') -> np.ndarray:
  use_gpu = torch.cuda.is_available()
  if use_gpu:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  geometry = read_xyz_file(geometry_path)
  R = np.array([[atom.x, atom.y, atom.z] for atom in geometry]) * 1.8897261258369282 # convert angstroms to bohr
  R = torch.stack([torch.tensor(R, dtype=torch.float32)]).to(device)
  output = model(R=R)
  F = output['full_hamiltonian'][0].detach().cpu().numpy()
    
  # sort fock matrix back
  atoms = ''
  for atom in geometry:
    atoms += atom.type
  orbital_convention = 'fulvene_minimal_basis'
  F = transform_hamiltonians_from_lm_to_ao(F, atoms=atoms, convention=orbital_convention)

  return F
  
def infer_orbitals_from_F_model(model_path: str, 
                                geometry_path: str,
                                basis_set_size: int = 36,
                                cutoff=5.0) -> np.ndarray:
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # transform geometry into input batch
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=cutoff), dtype=torch.float32, device=device)
  input = converter(atoms)
  
  # load model
  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  # predicting Fock matrix
  output = model(input)
  values = output['F'].detach().cpu().numpy()
  F = values.reshape(basis_set_size, basis_set_size)
  F = 0.5 * (F + F.T)

  return F


def infer_orbitals_from_mo_model(model_path: str, 
                                geometry_path: str,
                                basis_set_size: int = 36,
                                cutoff=5.0) -> np.ndarray:
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # transform geometry into input batch
  atoms = io.read(geometry_path)
  converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=cutoff), dtype=torch.float32, device=device)
  input = converter(atoms)
  
  # load model
  model = torch.load(model_path, map_location=device).to(device)
  model.eval()

  # predicting MO coefficient C matrix
  output = model(input)
  for key in ['mo_coeffs', 'mo_coeffs_adjusted']:
    if key in output.keys():
      values = output[key].detach().cpu().numpy()
      mo = values.reshape(basis_set_size, basis_set_size)
      mo = np.asarray(mo.tolist(), order='C')
      return mo