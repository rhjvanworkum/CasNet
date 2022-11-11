from typing import Callable, Dict, Optional, Sequence, Union
import pytorch_lightning as pl
import torch
from torch import nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.nn as snn

class Hamiltonian(nn.Module):

    def __init__(
        self,
        output_key: str,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        aggregation_mode: str = 'sum',
        activation: Callable = F.silu,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            activation: activation function
            polarizability_key: the key under which the predicted polarizability will be stored
        """
        super(Hamiltonian, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.output_key = output_key
        self.model_outputs = [output_key]

        self.aggregation_mode = aggregation_mode

        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs):
        # positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]
        # dim = l1.shape[-2]

        l0, l1 = self.outnet((l0, l1))

        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            l0 = snn.scatter_add(l0, idx_m, dim_size=maxm)
            l0 = torch.squeeze(l0, -1)

            if self.aggregation_mode == "avg":
                l0 = l0 / inputs[properties.n_atoms]

        inputs[self.output_key] = l0
        return inputs

class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        basis_set_size: int = 36,
        metrics: Optional[Dict[str, Metric]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.basis_set_size = basis_set_size
        self.metrics = nn.ModuleDict(metrics)
        self.constraints = []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property], self.basis_set_size
        )
        return loss

    def calculate_metrics(self, pred, target):
        metrics = {
            metric_name: metric(pred[self.name], target[self.target_property])
            for metric_name, metric in self.metrics.items()
        }
        return metrics