import math
import torch
import torch.nn as nn
import e3nn
from e3nn.o3 import Irreps
from torch_scatter import scatter




class SO3Convolution(nn.Module):
    """
    SO3-equivariant convolution using Clebsch-Gordon tensor product.
    With combined indexing s=(l,m), this can be written as:

    """

    def __init__(self, irreps_sh: Irreps, irreps_input: Irreps, irreps_output: Irreps):
        super().__init__()
        self.irreps_sh = irreps_sh
        self.irreps_input = irreps_input
        self.irreps_output = irreps_output

        self.tp = e3nn.o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)
        
        radial_dim = 12
        self.filter_network = e3nn.nn.FullyConnectedNet([radial_dim, self.tp.weight_numel // 2, self.tp.weight_numel], torch.relu)

        
    def forward(self, node_features: torch.Tensor,
                      edge_sh_features: torch.Tensor,
                      edge_radial_features: torch.Tensor,
                      edge_src: torch.Tensor,
                      edge_dst: torch.Tensor,
                      n_nodes: int):
        n_nodes
        tensor_product = self.tp(node_features[..., edge_src, :], 
                                 edge_sh_features, 
                                 self.filter_network(edge_radial_features))
        return scatter(tensor_product, edge_dst, dim=-2, dim_size=n_nodes, reduce="sum").div((n_nodes - 1)**0.5)
        