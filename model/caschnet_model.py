from typing import Callable
import torch
import schnetpack as spk
from model.architecture.model_output import ModelOutput, Hamiltonian

from torch.optim.lr_scheduler import _LRScheduler
class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

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
    pred_module = Hamiltonian(
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
        # scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        # scheduler_args={'threshold': 1e-6, 'patience': 5},
        scheduler_cls=NoamLR,
        scheduler_args={'warmup_steps': 20},
        scheduler_monitor='val_loss'
    )

    return task