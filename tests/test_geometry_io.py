"""Tests for XYZ geometry I/O and interpolation (pure-Python, NumPy only)."""

import numpy as np

from casnet.data.utils import (
    Atom,
    get_pos_matrix,
    interpolate_geometry,
    read_xyz_file,
    write_xyz_file,
)


def _water() -> list[Atom]:
    return [
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.75800, 0.58700, 0.0),
        Atom("H", -0.75800, 0.58700, 0.0),
    ]


def test_atom_coordinates():
    atom = Atom("C", 1.0, -2.0, 3.5)
    assert np.allclose(atom.coordinates, [1.0, -2.0, 3.5])


def test_write_read_roundtrip(tmp_path):
    atoms = _water()
    path = tmp_path / "water.xyz"
    write_xyz_file(atoms, str(path))

    loaded = read_xyz_file(str(path))
    assert [a.type for a in loaded] == [a.type for a in atoms]
    assert np.allclose(get_pos_matrix(loaded), get_pos_matrix(atoms), atol=1e-5)


def test_xyz_header_atom_count(tmp_path):
    path = tmp_path / "water.xyz"
    write_xyz_file(_water(), str(path))
    with open(path) as f:
        first_line = f.readline()
    assert int(first_line) == 3


def test_negative_and_positive_coords_roundtrip(tmp_path):
    atoms = [Atom("C", -1.23456, 2.34567, -3.45678)]
    path = tmp_path / "c.xyz"
    write_xyz_file(atoms, str(path))
    loaded = read_xyz_file(str(path))
    assert np.allclose(loaded[0].coordinates, atoms[0].coordinates, atol=1e-5)


def test_interpolate_geometry_endpoints_and_count():
    g0 = [Atom("C", 0.0, 0.0, 0.0), Atom("H", 1.0, 0.0, 0.0)]
    g1 = [Atom("C", 0.0, 0.0, 0.0), Atom("H", 3.0, 0.0, 0.0)]
    geoms = interpolate_geometry(g0, g1, 5)

    assert len(geoms) == 5
    # endpoints preserved
    assert np.isclose(geoms[0][1].x, 1.0)
    assert np.isclose(geoms[-1][1].x, 3.0)
    # midpoint is the linear average
    assert np.isclose(geoms[2][1].x, 2.0)
    # atom types carried over
    assert [a.type for a in geoms[0]] == ["C", "H"]
