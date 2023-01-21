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

mo_indices = [21, 23]

atom_dict = {
  'C': 6,
  'H': 1
}

cpk_colors = {
  6: [0.5, 0.5 ,0.5],
  1: [1, 1, 1]
}

def write_orbitals_to_molden(
  molden_file: str, 
  geometry_file: str,
  basis: str,
  orbitals: np.ndarray, 
  energies: np.ndarray
) -> None:

  molecule = gto.M(atom=geometry_file,
                   basis=basis,
                   spin=0,
                   symmetry=True)

  with open(molden_file, 'w') as f:
      molden.header(molecule, f)
      molden.orbital_coeff(molecule, f, orbitals, ene=energies)

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
  name: str, 
  geometry_file: str,
  basis: str,
  orbitals: np.ndarray, 
  energies: np.ndarray
):
  mol = gto.M(atom=geometry_file,
                  basis=basis,
                  spin=0,
                  symmetry=True)

  geometry = read_xyz_file(geometry_file)
  R = np.array([[atom.x, atom.y, atom.z] for atom in geometry]) / 0.529177249
  Z = np.array([atom_dict[atom.type] for atom in geometry])

  x, y, z, grid_coords = construct_cartesian_grid(R, box_margin, resolution)

  fig = mlab.figure(size=(500, 500), bgcolor=(0.95, 0.95, 0.95))
  mlab.view(azimuth=180, elevation=140, distance=None)

  for mo_idx in mo_indices:
    for coords, atom in zip(R, Z):
        mlab.points3d(
            *coords,
            scale_factor=covalent_radii[atom] * 2,
            resolution=resolution,
            color=tuple(cpk_colors[atom])
        )

    orb = np.einsum('ij,j->i', mol.eval_gto('GTOval', grid_coords), orbitals[:, mo_idx])
    orb = orb.reshape(x.shape)
    
    if np.min(orb) > -ORBITAL_CONTOUR:
        contours = [ORBITAL_CONTOUR]
    else:
        contours = [-ORBITAL_CONTOUR, ORBITAL_CONTOUR]
    
    mlab.contour3d(x, y, z, orb, 
                contours=contours, transparent=False, opacity=0.6, 
                colormap='blue-red', vmin=-ORBITAL_CONTOUR, vmax=ORBITAL_CONTOUR)
    
    f = mlab.gcf()
    f.scene.render_window.point_smoothing = True
    f.scene.render_window.line_smoothing = True
    f.scene.render_window.polygon_smoothing = True
    f.scene.render_window.multi_samples = 8 # Try with 4 if you think this is slow
    mlab.savefig(f'results/{name}_{mo_idx}.png')
    mlab.clf()


if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  parser = argparse.ArgumentParser()
  parser.add_argument('--geometry_path', type=str)
  parser.add_argument('--mode', type=str)
  parser.add_argument('--model', type=str)
  parser.add_argument('--basis', type=str)
  parser.add_argument('--name', type=str)
  args = parser.parse_args()

  geometry_path = base_dir + args.geometry_path
  mode = args.mode
  model_path = './checkpoints/' + args.model + '.pt'
  basis = args.basis

  method = initial_guess_dict[mode]
  mo_e, mo = method(model_path, geometry_path, basis)
  
  plot_orbital(
    # 'results/' + geometry_path.split('/')[-1].split('.')[0] + '_' + '.molden', 
    args.name,
    geometry_path,
    basis,
    mo,
    mo_e
  )

  # write_orbitals_to_molden(
#     # 'results/' + geometry_path.split('/')[-1].split('.')[0] + '_' + '.molden', 
#     f'results/{args.name}.molden',
#     geometry_path,
#     basis,
#     mo,
#     mo_e
# )