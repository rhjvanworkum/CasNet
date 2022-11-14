import e3nn
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear

from model.architecture.so3_activations import ShiftedSoftPlus
from model.architecture.so3_convolution import SO3Convolution


class SO3InteractionBlock(nn.Module):

    def __init__(self, input_irreps: Irreps, 
                       output_irreps: Irreps, 
                       sh_irreps: Irreps,
                       use_residual_connections: bool,
                       device: torch.device) -> None:
        super().__init__()
        self.input_irreps = input_irreps
        self.output_irreps = output_irreps
        self.sh_irreps = sh_irreps
        self.use_residual_connections = use_residual_connections

        self.linear_1 = Linear(
            irreps_in=self.input_irreps,
            irreps_out=self.input_irreps,
            internal_weights=True,
            shared_weights=True,
        ).to(device)

        self.convolution = SO3Convolution(
            irreps_sh=self.sh_irreps,
            irreps_input=self.input_irreps,
            irreps_output=self.output_irreps
        ).to(device)

        self.equivariant_nonlin = e3nn.nn.NormActivation(
            irreps_in=self.convolution.irreps_output,
            scalar_nonlinearity=ShiftedSoftPlus,
            normalize=True,
            epsilon=1e-8,
            bias=False,
        ).to(device)

        self.linear_2 = Linear(
            irreps_in=self.convolution.irreps_output,
            irreps_out=self.convolution.irreps_output,
            internal_weights=True,
            shared_weights=True,
        ).to(device)

        if self.use_residual_connections:
            self.residual_tp = e3nn.o3.FullyConnectedTensorProduct(
                self.input_irreps,
                self.input_irreps,
                self.output_irreps,
            ).to(device)

    def forward(self, x: torch.Tensor, edge_radial: torch.Tensor, edge_sh: torch.Tensor,
                      edge_src: torch.Tensor, edge_dst: torch.Tensor, n_nodes: int):
        out = self.linear_1(x)
        out = self.convolution(out, edge_sh, edge_radial,
                               edge_src, edge_dst, n_nodes)
        out = self.linear_2(out)
        out = self.equivariant_nonlin(out)
        if self.use_residual_connections:
            residual = self.residual_tp(x, x)
            out = residual + out
        return out