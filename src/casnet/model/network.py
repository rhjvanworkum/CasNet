"""Build the SchNetPack orbital-prediction model (PaiNN backbone + matrix output)."""

from collections.abc import Callable

import schnetpack as spk
import torch
from schnetpack import ModelOutput

from casnet.config import DEFAULT_BASIS_SET_SIZE, DEFAULT_CUTOFF
from casnet.model.architecture.outputs import HamiltonianOutput, MatrixOutput


def create_orbital_model(
    loss_function: Callable,
    lr: float = 5e-4,
    output_property_key: str = "F",
    basis_set_size: int = DEFAULT_BASIS_SET_SIZE,
    cutoff: float = DEFAULT_CUTOFF,
):

    pairwise_distance = spk.atomistic.PairwiseDistances()
    representation = spk.representation.PaiNN(
        n_atom_basis=64,
        n_interactions=5,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )
    if output_property_key == "F":
        pred_module = HamiltonianOutput(
            output_key=output_property_key,
            n_in=representation.n_atom_basis,
            n_layers=2,
            n_out=basis_set_size**2,
            basis_set_size=basis_set_size,
        )
    else:
        pred_module = MatrixOutput(
            output_key=output_property_key,
            n_in=representation.n_atom_basis,
            n_layers=2,
            n_out=basis_set_size**2,
            basis_set_size=basis_set_size,
        )
    nnp = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_module],
    )

    output = ModelOutput(
        name=output_property_key, loss_fn=loss_function, loss_weight=1.0, metrics={}
    )

    # Putting it in the Atomistic Task framework
    task = spk.AtomisticTask(
        model=nnp,
        outputs=[output],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={"threshold": 1e-6, "patience": 5},
        scheduler_monitor="val_loss",
    )

    return task
