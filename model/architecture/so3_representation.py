from typing import Callable, Dict, Optional

import hydra
import torch
import torch.nn as nn

import schnetpack.nn as snn
import schnetpack.nn.so3 as so3
import schnetpack.properties as properties

# class FactorizedConvolution(Module, torch.nn.Module):
#     avg_num_neighbors: Optional[float]
#     use_sc: bool

#     def __init__(
#         self,
#         input_features,
#         output_features,
#         node_attrs,
#         edge_radial,
#         edge_spherical,
#         invariant_layers=1,
#         invariant_neurons=8,
#         avg_num_neighbors=None,
#         use_sc=True,
#         nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
#         reduce=True,
#     ) -> None:
#         super().__init__()

#         self.init_irreps(
#             input_features=input_features,
#             output_features=output_features,
#             node_attrs=node_attrs,
#             edge_radial=edge_radial,
#             edge_spherical=edge_spherical,
#             output_keys=["output_features"],
#         )

#         self.avg_num_neighbors = avg_num_neighbors
#         self.use_sc = use_sc

#         feature_irreps_in = self.irreps_in["input_features"]
#         feature_irreps_out = self.irreps_out["output_features"]
#         irreps_edge_attr = self.irreps_in["edge_spherical"]

#         # - Build modules -
#         self.linear_1 = Linear(
#             irreps_in=feature_irreps_in,
#             irreps_out=feature_irreps_in,
#             internal_weights=True,
#             shared_weights=True,
#         )

#         # TODO: remove this
#         feature_irreps_out = Irreps(feature_irreps_out)

#         self.tp = TensorProductExpansion(
#             feature_irreps_in,
#             (irreps_edge_attr, "edge_spherical"),
#             (feature_irreps_out, "edge_features"),
#             "uvu",
#             internal_weight=False,
#         )

#         # init_irreps already confirmed that the edge embeddding is all invariant scalars
#         self.fc = FullyConnectedNet(
#             [Irreps(self.irreps_in["edge_radial"]).num_irreps]
#             + invariant_layers * [invariant_neurons]
#             + [self.tp.tp.weight_numel],
#             activations["ssp"],
#         )

#         self.sc = None
#         if self.use_sc:
#             self.sc = FullyConnectedTensorProduct(
#                 feature_irreps_in,
#                 Irreps(self.irreps_in["node_attrs"]),
#                 feature_irreps_out,
#             )

#         self.reduce = reduce

#     def forward(self, data: Dict[str, Tensor], attrs:Dict[str, Tuple[str, str]]):
#         input = data
#         weight = self.fc(input["edge_radial"])

#         x = input["input_features"]
#         edge_src = input["edge_index"][0]
#         edge_dst = input["edge_index"][1]

#         if self.sc is not None:
#             sc = self.sc(x, input["node_attrs"])

#         x = self.linear_1(x)
        
#         edge_features = self.tp(
#             left=x[edge_src], right=input["edge_spherical"], weight=weight
#         )
#         if self.reduce:
#             # [edges, feature_dim], [edges, sh_dim], [edges, weight_numel]
#             x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

#             # Necessary to get TorchScript to be able to type infer when its not None
#             avg_num_neigh: Optional[float] = self.avg_num_neighbors
#             if avg_num_neigh is not None:
#                 x = x.div(avg_num_neigh ** 0.5)

#             if self.sc is not None:
#                 x = x + sc
#         else:
#             x = edge_features

#         is_per = attrs["input_features"][0]
#         attrs = {"output_features": (is_per, self.irreps_out["output_features"])}
#         data = {"output_features": x}
#         return data, attrs



class SO3net(nn.Module):
    """
    A simple SO3-equivariant representation using spherical harmonics and
    Clebsch-Gordon tensor products.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        lmax: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        shared_interactions: bool = False,
        max_z: int = 100,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            lmax: maximum angular momentum of spherical harmonics basis
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            shared_interactions:
            max_z:
            conv_layer:
        """
        super(SO3net, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.lmax = lmax
        # self.cutoff_fn = hydra.utils.instantiate(cutoff_fn)
        # self.cutoff = cutoff_fn.cutoff
        # self.radial_basis = hydra.utils.instantiate(radial_basis)
        self.cutoff_fn = cutoff_fn
        self.cutoff = 5.0
        self.radial_basis = radial_basis
        
        # node embedding
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.sh = None
        
        self.so3convs = snn.replicate_module(
            lambda: so3.SO3Convolution(lmax, n_atom_basis, self.radial_basis.n_rbf),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings1 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings2 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings3 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.gatings = snn.replicate_module(
            lambda: so3.SO3ParametricGatedNonlinearity(n_atom_basis, lmax),
            self.n_interactions,
            shared_interactions,
        )
        self.so3product = so3.SO3TensorProduct(lmax)
        
    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        
        # Yij = self.sphharm(dir_ij)
        # radial_ij = self.radial_basis(d_ij)
        # cutoff_ij = self.cutoff_fn(d_ij)[..., None]

        x0 = self.embedding(atomic_numbers)[:, None]
        # shape [batch_size x n_atoms, l_channels, n_atom_basis]
        x = so3.scalar2rsh(x0, self.lmax) 

        # for i in range(self.n_interactions):
        #     dx = self.so3convs[i](x, radial_ij, Yij, cutoff_ij, idx_i, idx_j)
        #     ddx = self.mixings1[i](dx)
        #     dx = self.so3product(dx, ddx)
        #     dx = self.mixings2[i](dx)
        #     dx = self.gatings[i](dx)
        #     dx = self.mixings3[i](dx)
        #     x = x + dx

        # inputs["scalar_representation"] = x[:, 0]
        # inputs["multipole_representation"] = x
        # return inputs
