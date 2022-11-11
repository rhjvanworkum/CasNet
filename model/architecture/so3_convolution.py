import math
import torch
import torch.nn as nn
import schnetpack.nn as snn
import o3

class SO3Convolution(nn.Module):
    """
    SO3-equivariant convolution using Clebsch-Gordon tensor product.
    With combined indexing s=(l,m), this can be written as:

    """

    def __init__(self, lmax: int, n_atom_basis: int, n_radial: int):
        self.lmax = lmax
        
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)
        
        tp = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_output, shared_weights=False)
        
        
    def forward(self, input_features,):
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        