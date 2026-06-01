"""Molden file output for CASSCF orbitals.

Kept separate from :mod:`casnet.data.utils` because it depends on PySCF, which
is an optional (quantum-chemistry) dependency. Importing this module therefore
requires the ``qc`` extra to be installed.
"""

from __future__ import annotations

import numpy as np


def write_orbitals_to_molden(
    molden_file: str,
    geometry_file: str,
    basis: str,
    orbitals: np.ndarray,
    energies: np.ndarray,
) -> None:
    """Write orbital coefficients and energies to a Molden file.

    Args:
        molden_file: Output path for the Molden file.
        geometry_file: Path to the molecular geometry (XYZ).
        basis: Basis-set name understood by PySCF.
        orbitals: MO coefficient matrix.
        energies: MO energies.
    """
    from pyscf import gto
    from pyscf.tools import molden

    molecule = gto.M(atom=geometry_file, basis=basis, spin=0, symmetry=True)

    with open(molden_file, "w") as f:
        molden.header(molecule, f)
        molden.orbital_coeff(molecule, f, orbitals, ene=energies)
