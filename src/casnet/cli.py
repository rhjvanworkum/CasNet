"""Command-line entry points for CasNet.

Exposes the ``casnet-train`` console script (see ``pyproject.toml``). This
replaces the previous top-level ``train_model.py`` script; behaviour is
preserved, but data/checkpoint locations are now resolved via
:mod:`casnet.config` (overridable with environment variables) instead of being
hard-coded relative paths.
"""

from __future__ import annotations

import argparse
import os

import torch

from casnet.config import (
    DEFAULT_BASIS_SET_SIZE,
    DEFAULT_CUTOFF,
    get_checkpoints_dir,
    get_data_storage_dir,
)
from casnet.model.network import create_orbital_model
from casnet.model.training import train_model


def train() -> None:
    """``casnet-train`` entry point: train an ML-MO or ML-F orbital model."""
    epochs = 100
    lr = 1e-3
    batch_size = 16

    parser = argparse.ArgumentParser(description=train.__doc__)
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--property", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--basis_set_size", type=int, default=DEFAULT_BASIS_SET_SIZE)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    data_storage = get_data_storage_dir()
    database_path = os.path.join(data_storage, args.db_name)
    split_file = os.path.join(data_storage, args.split_name)
    save_path = os.path.join(get_checkpoints_dir(), args.model_name + ".pt")

    train_model(
        save_path=save_path,
        property=args.property,
        loss_fn=torch.nn.functional.mse_loss,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        basis_set_size=args.basis_set_size,
        database_path=database_path,
        create_model_fn=create_orbital_model,
        split_file=split_file,
        use_wandb=args.use_wandb,
        cutoff=args.cutoff,
    )


if __name__ == "__main__":
    train()
