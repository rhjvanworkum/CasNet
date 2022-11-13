from typing import Callable, Dict, Optional, Sequence, Union
import pytorch_lightning as pl
import torch
from torch import nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

import schnetpack as spk
import schnetpack.properties as properties
import schnetpack.nn as snn

class HamiltonianOutput(nn.Module):

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
        super(HamiltonianOutput, self).__init__()
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
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]

        l0, l1 = self.outnet((l0, l1))

        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            l0 = snn.scatter_add(l0, idx_m, dim_size=maxm)
            l0 = torch.squeeze(l0, -1)

            if self.aggregation_mode == "avg":
                l0 = l0 / inputs[properties.n_atoms]

        H = l0.reshape(-1, 14, 14)
        H = H + torch.transpose(H, -1, -2)
        H = H.reshape(-1)

        inputs[self.output_key] = H
        return inputs