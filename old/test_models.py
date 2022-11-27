import os
from typing import List
import argparse
import numpy as np
from data.db.save_casscf_calculations_to_db import get_orbital_order
from pyscf import gto, mcscf, scf
from functools import reduce
from ase.db import connect
import scipy
import scipy.linalg
import torch
import schnetpack as spk
from ase import io


mo_scores = []
f_scores = []
dm_scores = []

def get_orbitals_from_f(F, S):
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo

def get_orbitals_from_dm(casscf, dm, S):
  vj, vk = casscf._scf.get_jk(casscf.mol, dm)
  F = casscf.get_hcore() + vj-vk*.5
  mo_e, mo = scipy.linalg.eigh(F, S)
  return mo

n = 36

if __name__ == "__main__":
  for i in range(10):
    idx = np.random.choice(np.arange(200), 1)[0]
    print(idx)
    # geometry_file_path = f"/mnt/c/users/rhjva/imperial/fulvene/geometries/ethene_geom_scan/geometry_{idx}.xyz"
    geometry_file_path = f"/mnt/c/users/rhjva/imperial/fulvene/geometries/geom_scan_200/geometry_{idx}.xyz"
    molecule = gto.M(atom=geometry_file_path,
                    basis='sto_6g',
                    spin=0,
                    symmetry=True)

    hartree_fock = molecule.RHF()
    # hartree_fock.kernel()
    S = hartree_fock.get_ovlp(molecule)

    n_states = 2
    weights = np.ones(n_states) / n_states
    casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
    casscf.conv_tol = 1e-8

    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    atoms = io.read(geometry_file_path)
    converter = spk.interfaces.AtomsConverter(neighbor_list=spk.transform.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device)
    input = converter(atoms)
    
    # model 1
    model = torch.load('checkpoints/fulvene_mo_test_3states.pt', map_location=device).to(device)
    model.eval()
    output = model(input)
    values = output['mo_coeffs_adjusted'].detach().cpu().numpy()[0]
    mos = np.asarray(values.reshape(36, 36).tolist(), order='C')
    conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mos)
    mo_scores.append((imacro, imicro, iinner))

    # model 2
    model = torch.load('checkpoints/fulvene_f_test_3states.pt', map_location=device).to(device)
    model.eval()
    output = model(input)
    values = output['F'].detach().cpu().numpy()[0]
    F = np.asarray(values.reshape(36, 36).tolist(), order='C')
    F = 0.5 * (F + F.T)
    mos = get_orbitals_from_f(F, S)
    conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mos)
    f_scores.append((imacro, imicro, iinner))

    # model 3
    model = torch.load('checkpoints/fulvene_dm_test_3states.pt', map_location=device).to(device)
    model.eval()
    output = model(input)
    values = output['dm'].detach().cpu().numpy()[0]
    dm = np.asarray(values.reshape(36, 36).tolist(), order='C')
    mos = get_orbitals_from_dm(casscf, dm, S)
    conv, e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(mos)
    dm_scores.append((imacro, imicro, iinner))

for scores in [f_scores, mo_scores, dm_scores]:
 print(f'scores: {np.mean(np.array([val[0] for val in scores]))}, {np.mean(np.array([val[1] for val in scores]))}, {np.mean(np.array([val[2] for val in scores]))},')

