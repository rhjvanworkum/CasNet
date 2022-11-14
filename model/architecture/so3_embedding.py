import torch
import torch.nn as nn
import e3nn

from .utils import PolynomialCutoff, BesselBasis

class OneHotEncoding(nn.Module):
    def __init__(self, z_max: int, 
                       n_embedding: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(z_max + 1, n_embedding, padding_idx=0)

    def forward(self, features: torch.Tensor):
        """
        :param features: tensor containing input features to embed
        """
        return self.embedding(features)

class SphericalHarmonicEncoding(nn.Module):
    def __init__(self, lmax: int, 
                       normalize: bool = True,
                       normalization: str = "component") -> None:
        super().__init__()
        self.lmax = lmax
        self.normalize = normalize
        self.normalization = normalization
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax=lmax)

    def forward(self, features: torch.Tensor):
        """
        :param features: tensor containing input features, usually distance vectors
        """
        return e3nn.o3.spherical_harmonics(self.irreps_sh,
                                           features, 
                                           normalize=self.normalize, 
                                           normalization=self.normalization)

class RadialBasisEncoding(nn.Module):
    def __init__(self, r_max: int,
                       r_min: int = 0,
                       num_basis: int = 12) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.r_min = r_min
        self.r_max = r_max
        # self.basis = BesselBasis(r_max, r_min, num_basis, trainable=True, one_over_r=True)
        # self.cutoff = PolynomialCutoff(r_max, p=6)

    def forward(self, features: torch.Tensor):
        """
        :param features: tensor containing input features, usually distance vectors
        """
        return e3nn.math.soft_one_hot_linspace(features.norm(dim=-1), self.r_min, self.r_max, 
                                               self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)


class Embedding(nn.Module):
    def __init__(self, lmax: int, z_max: int, n_atom_basis: int, cutoff: float, n_radial_basis: int) -> None:
        super().__init__()

        # node embedding + edge vector embedding
        self.node_embedding = OneHotEncoding(z_max, n_atom_basis)
        self.edge_sh_embedding = SphericalHarmonicEncoding(lmax)
        self.edge_radial_embedding = RadialBasisEncoding(r_max=cutoff, num_basis=n_radial_basis)

    def forward(self, atomic_number: torch.Tensor, edge_vec: torch.Tensor):
        node_features = self.node_embedding(atomic_number)
        edge_radial_features = self.edge_radial_embedding(edge_vec)
        edge_sh_features = self.edge_sh_embedding(edge_vec)
        return node_features, edge_radial_features, edge_sh_features
