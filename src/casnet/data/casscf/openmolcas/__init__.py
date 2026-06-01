"""OpenMolcas backend configuration and input-file path helpers."""

from __future__ import annotations

from casnet.config import get_molcas_path

#: Path to the ``pymolcas`` executable, configurable via the ``MOLCAS_PATH`` env var.
MOLCAS_PATH = get_molcas_path()


def get_input_file(basis: str) -> str:
    return f"./data/files/input_{basis}.input"


def get_guess_orb_file(basis: str) -> str:
    return f"./data/files/orb_{basis}.orb"


def get_seward_input_file(basis: str) -> str:
    return f"./data/files/seward_{basis}.input"
