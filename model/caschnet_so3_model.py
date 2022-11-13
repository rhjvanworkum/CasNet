from typing import Callable
import torch
from model.architecture.so3_representation import SO3net
import schnetpack as spk
from schnetpack import ModelOutput

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

def create_so3_orbital_model(loss_function: Callable,
                         lr: float = 5e-4,
                         output_property_key: str = 'F',
                         basis_set_size: int = 36,
                         cutoff: float = 5.0):

    model = SO3net(
        n_atom_basis=12,
        n_radial_basis=12,
        n_interaction_layers=2,
        use_residual_connections=True,
        lmax=2
    )

    nnp = spk.model.NeuralNetworkPotential(
        representation=model
    )
    nnp.model_outputs = ['F']

    output = ModelOutput(
        name='F',
        loss_fn=loss_function,
        loss_weight=1.0,
        metrics={}
    )

    task = spk.AtomisticTask(
        model=nnp,
        outputs=[output],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={'threshold': 1e-6, 'patience': 5},
        # scheduler_cls=NoamLR,
        # scheduler_args={'warmup_steps': 20},
        scheduler_monitor='val_loss'
    )

    return task