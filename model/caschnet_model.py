from typing import Callable
import torch
import schnetpack as spk
from model.architecture.model_output import ModelOutput

def create_orbital_model(loss_function: Callable,
                         lr: float = 5e-4,
                         output_property_key: str = 'F',
                         basis_set_size: int = 36,
                         cutoff: float = 5.0):

    pairwise_distance = spk.atomistic.PairwiseDistances()
    representation = spk.representation.PaiNN(
        n_atom_basis=64,
        n_interactions=5,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_module = spk.atomistic.Atomwise(
        output_key=output_property_key,
        n_in=representation.n_atom_basis,
        n_layers=2,
        n_out=basis_set_size**2
    )
    nnp = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_module],
    )

    output = ModelOutput(
        name=output_property_key,
        loss_fn=loss_function,
        loss_weight=1.0,
        basis_set_size=basis_set_size
    )

    # Putting it in the Atomistic Task framework
    task = spk.AtomisticTask(
        model=nnp,
        outputs=[output],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={'threshold': 1e-6, 'patience': 10},
        scheduler_monitor='val_loss'
    )

    return task