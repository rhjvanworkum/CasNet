"""Pure-Python utilities for CASSCF results and molecular geometry I/O.

This module deliberately depends only on NumPy (and tqdm) so that it can be
imported and unit-tested without the heavy quantum-chemistry / machine-learning
stack. PySCF-dependent helpers live in :mod:`casnet.data.molden`.
"""

from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CASSCF calculation results
# ---------------------------------------------------------------------------


class CasscfResult:
    """Container for the outputs of a single CASSCF calculation."""

    def __init__(
        self,
        converged: bool,
        basis: str,
        e_tot: float,
        mo_energies: np.ndarray,
        mo_coeffs: np.ndarray,
        S: np.ndarray,
        F: np.ndarray,
        imacro: int,
        index: int | None = None,
    ) -> None:
        self.converged = converged
        self.basis = basis
        self.e_tot = e_tot
        self.mo_energies = mo_energies
        self.mo_coeffs = mo_coeffs
        self.S = S
        self.F = F
        self.imacro = imacro

        self.index = index
        self.mo_coeffs_adjusted: np.ndarray | None = None

    def store_as_npz(self, file: str) -> None:
        """Serialise the result to a ``.npz`` archive."""
        np.savez(
            file,
            converged=self.converged,
            basis=self.basis,
            e_tot=self.e_tot,
            mo_energies=self.mo_energies,
            mo_coeffs=self.mo_coeffs,
            S=self.S,
            F=self.F,
            imacro=self.imacro,
        )

    @classmethod
    def load_from_npz(cls, file: str) -> CasscfResult:
        """Load a result from a ``.npz`` archive, parsing the index from the filename."""
        index = parse_index_from_filename(file)

        data = np.load(file, allow_pickle=True)
        return cls(
            data["converged"],
            data["basis"],
            data["e_tot"],
            data["mo_energies"],
            data["mo_coeffs"],
            data["S"],
            data["F"],
            data["imacro"],
            index,
        )


def parse_index_from_filename(file: str) -> int | None:
    """Extract the trailing integer index from a ``*_<idx>.<ext>`` filename.

    Returns ``None`` if the filename does not match the expected pattern.
    """
    try:
        return int(os.path.basename(file).split(".")[0].split("_")[-1])
    except (ValueError, IndexError):
        return None


def find_all_geometry_files_in_folder(geometry_folder: str) -> list[str]:
    """Return all ``.xyz`` files found (recursively) under ``geometry_folder``."""
    geometry_files = []
    for _, _, files in os.walk(geometry_folder):
        for file in files:
            if ".xyz" in file:
                geometry_files.append(geometry_folder + file)
    return geometry_files


def sort_geometry_files_by_idx(geometry_files: list[str]) -> list[str]:
    """Sort geometry files by the integer index encoded in their filename.

    Raises ``ValueError`` if any filename does not encode an index.
    """

    def _index(file: str) -> int:
        idx = parse_index_from_filename(file)
        if idx is None:
            raise ValueError(f"Cannot parse index from filename: {file}")
        return idx

    return sorted(geometry_files, key=_index)


def get_pos_matrix(geom: list[Atom]) -> np.ndarray:
    """Stack atom coordinates into an ``(n_atoms, 3)`` array."""
    return np.array([[atom.x, atom.y, atom.z] for atom in geom])


def sort_geometry_files_by_distance(
    geometry_files: list[str], start_geometry_file: str
) -> tuple[list[str], list[int]]:
    """Greedily order geometries by nearest-neighbour distance from a start geometry.

    Returns the reordered file list and the list of original indices in that order.
    """
    start_geometry = read_xyz_file(start_geometry_file)
    geometries = [read_xyz_file(file) for file in geometry_files]
    pos_matrices = [get_pos_matrix(geometry) for geometry in geometries]
    indices = np.arange(len(geometry_files)).tolist()
    selected_idxs = []
    sorted_geometry_files = []

    current_geometry = get_pos_matrix(start_geometry)
    for _ in tqdm(range(len(geometry_files)), total=len(geometry_files)):
        distances = [
            np.sum(np.linalg.norm(pos_matrices[idx] - current_geometry)) for idx in indices
        ]
        selected_idx = indices[np.argmin(distances)]
        indices.remove(selected_idx)
        selected_idxs.append(selected_idx)
        current_geometry = get_pos_matrix(geometries[selected_idx])
        sorted_geometry_files.append(geometry_files[selected_idx])

    return sorted_geometry_files, selected_idxs


def find_all_files_in_output_folder(output_folder: str) -> list[CasscfResult]:
    """Load every ``.npz`` CASSCF result found under ``output_folder``."""
    file_list = []
    for _, _, files in os.walk(output_folder):
        for file in files:
            if ".npz" in file:
                file_list.append(output_folder + file)
    return [CasscfResult.load_from_npz(file) for file in file_list]


def check_and_create_folder(folder: str) -> None:
    """Create ``folder`` (and parents) if it does not already exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)


# ---------------------------------------------------------------------------
# Geometry I/O
# ---------------------------------------------------------------------------


class Atom:
    """A single atom: element symbol and Cartesian coordinates."""

    def __init__(self, type: str, x: float, y: float, z: float) -> None:
        self.type = type
        self.x = x
        self.y = y
        self.z = z

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


def write_xyz_file(atoms: list[Atom], filename: str) -> None:
    """Write a list of :class:`Atom` to an XYZ file."""
    with open(filename, "w") as f:
        f.write(str(len(atoms)) + " \n")
        f.write("\n")

        for atom in atoms:
            f.write(atom.type)
            for coord in ["x", "y", "z"]:
                if getattr(atom, coord) < 0:
                    f.write("         ")
                else:
                    f.write("          ")
                f.write(f"{getattr(atom, coord):.5f}")
            f.write("\n")

        f.write("\n")


def read_xyz_file(filename: str) -> list[Atom]:
    """Read an XYZ file into a list of :class:`Atom`."""
    atoms = []

    with open(filename) as f:
        n_atoms = int(f.readline())
        _ = f.readline()

        for _ in range(n_atoms):
            data = f.readline().replace("\n", "").split(" ")
            data = list(filter(lambda a: a != "", data))
            atoms.append(Atom(data[0], float(data[1]), float(data[2]), float(data[3])))

    return atoms


def phase_correct_orbitals(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Flip the sign of each target orbital whose overlap with the reference is negative.

    Used to keep orbital phases consistent along a sequence of geometries before
    storing MO coefficients in the database.
    """
    ordered_target = target.copy()
    for idx in range(ordered_target.shape[0]):
        if np.dot(ref[:, idx], ordered_target[:, idx]) < 0:
            ordered_target[:, idx] = -1 * ordered_target[:, idx]
    return ordered_target


def interpolate_geometry(geom1: list[Atom], geom2: list[Atom], n: int) -> list[list[Atom]]:
    """Linearly interpolate ``n`` geometries between ``geom1`` and ``geom2``."""
    geometries = [[Atom("None", 0, 0, 0) for _ in range(len(geom1))] for _ in range(n)]

    for idx, (atom1, atom2) in enumerate(zip(geom1, geom2, strict=False)):
        atom_type = atom1.type
        x_coordinates = np.linspace(atom1.x, atom2.x, n)
        y_coordinates = np.linspace(atom1.y, atom2.y, n)
        z_coordinates = np.linspace(atom1.z, atom2.z, n)

        for i in range(n):
            geometries[i][idx].type = atom_type
            geometries[i][idx].x = x_coordinates[i]
            geometries[i][idx].y = y_coordinates[i]
            geometries[i][idx].z = z_coordinates[i]

    return geometries
