from evaluation.pyscf import initial_guess_dict
from pyscf import gto
from pyscf.tools import molden
import numpy as np
from typing import List, Tuple
import argparse
from mayavi import mlab
from ase.data import covalent_radii
import os

from data.utils import read_xyz_file

ORBITAL_CONTOUR = 0.02
box_margin = 4
resolution = 100

mo_indices = [17, 18, 19, 20]

atom_dict = {
  'C': 6,
  'H': 1
}

cpk_colors = {
  6: [0.5, 0.5 ,0.5],
  1: [1, 1, 1]
}

def get_bound_box(
    R: np.ndarray,
    box_margin: float
) -> List[Tuple[float]]:
    """
    Returns a rectangular box around a molecule, described by R,
    with margins.
    """
    return [(
        float(min(R[:, i])) - box_margin, 
        float(max(R[:, i])) + box_margin
    ) for i in range(3)]

def construct_cartesian_grid(
    R: np.ndarray,
    box_margin: float,
    resolution: int,
) -> np.ndarray:
    """
    Constructs a regular grid in cartesian space around a molecule,
    described by R.
    """
    (x1, x2), (y1, y2), (z1, z2) = get_bound_box(R, box_margin)
    nx, ny, nz = resolution * 1j, resolution * 1j, resolution * 1j
    x, y, z = np.mgrid[x1:x2:nx, y1:y2:ny, z1:z2:nz]
    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    return x, y, z, coords

def plot_orbital(
  output_dir: str,
  name: str, 
  geometry_file: str,
  basis: str,
  orbitals: np.ndarray
):
  mol = gto.M(atom=geometry_file,
                  basis=basis,
                  spin=0,
                  symmetry=True)

  geometry = read_xyz_file(geometry_file)
  R = np.array([[atom.x, atom.y, atom.z] for atom in geometry]) / 0.529177249
  Z = np.array([atom_dict[atom.type] for atom in geometry])

  x, y, z, grid_coords = construct_cartesian_grid(R, box_margin, resolution)

  mlab.options.offscreen = True

  fig = mlab.figure(size=(500, 500), bgcolor=(1,1,1))
  mlab.view(azimuth=180, elevation=140, distance=None)

  for mo_idx in mo_indices:
    for flip_idx in [-1, 1]:
      for coords, atom in zip(R, Z):
          mlab.points3d(
              *coords,
              scale_factor=covalent_radii[atom] * 2,
              resolution=resolution,
              color=tuple(cpk_colors[atom])
          )

      orb = np.einsum('ij,j->i', mol.eval_gto('GTOval', grid_coords), flip_idx * orbitals[:, mo_idx])
      orb = orb.reshape(x.shape)
      
      contours = []
      if np.min(orb) < -ORBITAL_CONTOUR:
        contours.append(-ORBITAL_CONTOUR)
      if np.max(orb) > ORBITAL_CONTOUR:
        contours.append(ORBITAL_CONTOUR)
      
      contour = mlab.contour3d(x, y, z, orb, 
                  contours=contours, transparent=False, opacity=1.0, 
                  colormap='blue-red', vmin=-ORBITAL_CONTOUR, vmax=ORBITAL_CONTOUR)
      contour.actor.property.interpolation = 'phong'
      contour.actor.property.specular = 0.1
      contour.actor.property.specular_power = 5
      
      mlab.savefig(os.path.join(output_dir, f'{name}_{mo_idx}_{flip_idx}.png'))
      mlab.clf()

if __name__ == "__main__":
  geometry_path = os.environ['base_dir'] + "geometries/fulvene_md_traj_25/geometry_13.xyz"

  # mo_model = f'./checkpoints/geom_scan_250/fulvene_gs250_MO.pt'
  # f_model = f'./checkpoints/geom_scan_250/fulvene_gs250_F.pt'
  # phisnet_model = f'./checkpoints/geom_scan_250/fulvene_gs250_phisnet.pt'
  mo_model = f'./checkpoints/md_traj/fulvene_mdtraj_MO.pt'
  f_model = f'./checkpoints/md_traj/fulvene_mdtraj_F.pt'
  phisnet_model = f'./checkpoints/md_traj/fulvene_mdtraj_phisnet.pt'

  output_dir = './results/orbitals_new/'

  basis = 'sto_6g'

  for mode, model in zip(
    ['casscf', 'ML-MO', 'ML-F', 'PhiSNet'],
    [None, mo_model, f_model, phisnet_model]
  ):
    method = initial_guess_dict[mode]
    _, mo = method(model, geometry_path, basis)
    plot_orbital(output_dir, mode, geometry_path, basis, mo)