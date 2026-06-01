"""Centralised configuration constants and path helpers for CasNet.

Historically these values were scattered across the codebase as magic numbers
(``ncas=6``, ``sort_mo([19, ...])``) and machine-specific absolute paths baked
into ``env.sh``. They are collected here so the fulvene-specific assumptions are
visible in one place and the file-system paths can be overridden via environment
variables rather than edited in source.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Fulvene CASSCF active-space configuration
# ---------------------------------------------------------------------------
#: Number of active orbitals in the fulvene CAS(6,6) active space.
NCAS = 6
#: Number of active electrons.
NELECAS = 6
#: Number of states in the state-averaged CASSCF.
N_STATES = 3
#: Orbital indices passed to ``casscf.sort_mo`` for the fulvene active space.
SORT_MO_INDICES = [19, 20, 21, 22, 23, 24]
#: CASSCF/CASCI convergence tolerance.
CONV_TOL = 1e-8

# ---------------------------------------------------------------------------
# Model / representation defaults
# ---------------------------------------------------------------------------
#: Default neighbour-list cutoff in Angstrom.
DEFAULT_CUTOFF = 5.0
#: Default flattened basis-set size (number of basis functions) for fulvene.
DEFAULT_BASIS_SET_SIZE = 36
#: Number of basis functions per supported basis set.
BASIS_SET_SIZES = {
    "sto_6g": 36,
    "ANO-S-MB": 36,
    "cc-pVDZ": 114,
}

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
ANGSTROM_TO_BOHR = 1.8897261258369282
BOHR_TO_ANGSTROM = 0.529177249


# ---------------------------------------------------------------------------
# Path / environment helpers
# ---------------------------------------------------------------------------
def get_base_dir() -> str:
    """Return the experiment base directory.

    Reads the ``CASNET_BASE_DIR`` (preferred) or legacy ``base_dir`` environment
    variable, falling back to the current working directory. This replaces the
    previous behaviour of crashing with ``KeyError`` when ``base_dir`` was unset.
    """
    return os.environ.get("CASNET_BASE_DIR") or os.environ.get("base_dir") or os.getcwd()


def get_data_storage_dir() -> str:
    """Directory holding ASE databases and split files (``CASNET_DATA_STORAGE``)."""
    return os.environ.get("CASNET_DATA_STORAGE", "./data_storage")


def get_checkpoints_dir() -> str:
    """Directory holding trained-model checkpoints (``CASNET_CHECKPOINTS``)."""
    return os.environ.get("CASNET_CHECKPOINTS", "./checkpoints")


def get_molcas_path() -> str:
    """Path to the OpenMolcas ``pymolcas`` executable (``MOLCAS_PATH``)."""
    return os.environ.get("MOLCAS_PATH", "pymolcas")
