"""Training loop for the orbital-prediction model (PyTorch Lightning)."""

import logging
import os
from collections.abc import Callable

import pytorch_lightning
import schnetpack as schnetpack
import torch
from pytorch_lightning.loggers import WandbLogger

from casnet.config import DEFAULT_BASIS_SET_SIZE, DEFAULT_CUTOFF, get_checkpoints_dir
from casnet.model.losses import symm_matrix_mse
from casnet.model.network import create_orbital_model


def train_model(
    save_path: str,
    property: str = "F",
    loss_fn: Callable = symm_matrix_mse,
    batch_size: int = 16,
    lr: float = 5e-4,
    epochs: int = 100,
    basis_set_size: int = DEFAULT_BASIS_SET_SIZE,
    database_path: str | None = None,
    split_file: str | None = None,
    use_wandb: bool = False,
    create_model_fn=create_orbital_model,
    initial_model_path: str | None = None,
    cutoff: float = DEFAULT_CUTOFF,
):
    """Train an orbital-prediction model and save the best checkpoint to ``save_path``."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    """ Initializing a dataset """
    dataset = schnetpack.data.datamodule.AtomsDataModule(
        datapath=database_path,
        batch_size=batch_size,
        split_file=split_file,
        transforms=[
            schnetpack.transform.ASENeighborList(cutoff=cutoff),
            schnetpack.transform.CastTo32(),
        ],
        property_units={property: 1.0},
        num_workers=8,
        pin_memory=True,
        load_properties=[property],
    )

    """ Initiating the Model """
    model = create_model_fn(
        loss_function=loss_fn,
        lr=lr,
        output_property_key=property,
        basis_set_size=basis_set_size,
        cutoff=cutoff,
    )

    if initial_model_path is not None:
        state_dict = torch.load(initial_model_path).state_dict()
        for key in list(state_dict.keys()):
            state_dict["model." + key] = state_dict.pop(key)
        model.load_state_dict(state_dict)

    # callbacks for PyTorch Lightning Trainer
    logging.info("Setup trainer")
    callbacks = [
        schnetpack.train.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath=get_checkpoints_dir(),
            filename="{epoch:02d}",
            model_path=save_path,
        ),
        pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pytorch_lightning.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-6, patience=50, verbose=False, mode="min"
        ),
    ]

    args = {
        "callbacks": callbacks,
        "default_root_dir": "./test/",
        "max_epochs": epochs,
        "devices": 1,
    }

    if torch.cuda.is_available():
        args["accelerator"] = "gpu"

    if use_wandb:
        wandb_project = os.environ.get("WANDB_PROJECT", "casnet")
        logger = WandbLogger(project=wandb_project)
        args["logger"] = logger

    trainer = pytorch_lightning.Trainer(**args)

    logging.info("Start training")
    trainer.fit(model, datamodule=dataset)
