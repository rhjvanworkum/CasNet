import os
from data.utils import Atom, read_xyz_file, write_xyz_file
import copy
import numpy as np
import matplotlib.pyplot as plt


def mean_squared_displacement(equilibrium_geometry, geometry):
  msd = 0
  for i_atom in range(len(equilibrium_geometry)):
    msd += np.linalg.norm(geometry[i_atom].coordinates - equilibrium_geometry[i_atom].coordinates) ** 2
  return msd / len(equilibrium_geometry)

def plot_msd_bin_plot(equilibrium_geometry, geometries):
  msd = [mean_squared_displacement(equilibrium_geometry, geometry) for geometry in geometries]
  plt.hist(msd, density=True, bins=50)
  plt.savefig('./results/msd.png')


if __name__ == "__main__":
  base_dir = os.environ['base_dir']

  sigma = 1
  n = 200
  folder = base_dir + 'geometries/normal_dist_s1_200/'
  
  if not os.path.exists(folder):
    os.makedirs(folder)

  equilibrium_geometry = read_xyz_file('./data_storage/gs_fulvene_equilibrium.xyz')
  n_atoms = len(equilibrium_geometry)
  coords = ['x', 'y', 'z']
  geometries = []

  for idx in range(n):
    geometry = [Atom(a.type, a.x , a.y, a.z) for a in equilibrium_geometry]
    for i_atom in range(n_atoms):
      for j in range(3):
        displacement = np.random.normal(loc=0, scale=sigma, size=1)
        setattr(geometry[i_atom], coords[j], getattr(geometry[i_atom], coords[j]) + displacement)

    geometries.append(geometry)

  # plot bin plot
  plot_msd_bin_plot(equilibrium_geometry, geometries)

  # write geometries
  for idx, geometry in enumerate(geometries):
    write_xyz_file(geometry, f'{folder}geometry_{idx}.xyz')