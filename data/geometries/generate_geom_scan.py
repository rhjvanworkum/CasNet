from typing import List
import numpy as np

class Atom:
  def __init__(self, type, x, y, z) -> None:
    self.type = type
    self.x = x
    self.y = y
    self.z = z

def write_xyz_file(atoms: List[Atom], filename: str):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
      f.write(atom.type)
      for coord in ['x', 'y', 'z']:
        if getattr(atom, coord) < 0:
          f.write('         ')
        else:
          f.write('          ')
        f.write("%.5f" % getattr(atom, coord))
      f.write('\n')
    
    f.write('\n')

def read_xyz_file(filename):
  atoms = []

  with open(filename) as f:
    n_atoms = int(f.readline())
    _ = f.readline()

    for i in range(n_atoms):
      data = f.readline().replace('\n', '').split(' ')
      data = list(filter(lambda a: a != '', data))
      atoms.append(Atom(data[0], float(data[1]), float(data[2]), float(data[3])))

  return atoms

def interpolate_geometry(geom1, geom2, n):
  geometries = [[Atom('None', 0, 0, 0) for _ in range(len(geom1))] for _ in range(n)]

  for idx, (atom1, atom2) in enumerate(zip(geom1, geom2)):
    type = atom1.type
    x_coordinates = np.linspace(atom1.x, atom2.x, n)
    y_coordinates = np.linspace(atom1.y, atom2.y, n)
    z_coordinates = np.linspace(atom1.z, atom2.z, n)

    for i in range(n):
      geometries[i][idx].type = type
      geometries[i][idx].x = x_coordinates[i]
      geometries[i][idx].y = y_coordinates[i]
      geometries[i][idx].z = z_coordinates[i]

  return geometries


if __name__ == "__main__":
  fulvene_0 = read_xyz_file('/home/ruard/Documents/experiments/fulvene/geometries/geom_scan_200/geometry_0.xyz')
  fulvene_1 = read_xyz_file('/home/ruard/Documents/experiments/fulvene/geometries/geom_scan_200/geometry_199.xyz')

  geometries = interpolate_geometry(fulvene_0, fulvene_1, 250)

  for idx, geometry in enumerate(geometries):
    write_xyz_file(geometry, '/home/ruard/Documents/experiments/fulvene/geometries/fulvene_geometry_scan_250/geometry_' + str(idx) + '.xyz')

