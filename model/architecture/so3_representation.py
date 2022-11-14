from typing import Callable, Dict, Optional

import hydra
import torch
import torch.nn as nn
from model.architecture.clebsch_gordan import ClebschGordan
from model.architecture.so3_convolution import SO3Convolution
from model.architecture.so3_embedding import Embedding, OneHotEncoding
from model.architecture.so3_interaction_block import SO3InteractionBlock

import schnetpack.properties as properties
import numpy as np


import e3nn
from e3nn.o3 import Linear, Irreps
from torch_scatter import scatter
import itertools



def get_filter_possible_out_irreps(lmax: int) -> Callable:
    allowed_orders = [str(l) + p for (l, p) in itertools.product(range(lmax), ['e', 'o'])]
    filter_fn = lambda x: any(order in x for order in allowed_orders)
    return filter_fn

def determine_feature_output_irreps(lmax: int, input_irreps: Irreps, sh_irreps: Irreps) -> Irreps:
    tp = e3nn.o3.FullTensorProduct(
        irreps_in1=input_irreps,
        irreps_in2=sh_irreps,
    )
    possible_out_irreps = str(tp.irreps_out).split('+')
    irreps_output = '+'.join(filter(get_filter_possible_out_irreps(lmax), possible_out_irreps))
    return irreps_output

def generate_l_combs(l1, l2):
            lower = int(abs(l2 - l1))
            upper = int(l1 + l2)
            if lower == upper:
                return [lower]
            else:
                return np.arange(lower, upper + 1).tolist()




class TensorProductExpansion(nn.Module):

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.clebsch_gordan = ClebschGordan().to(device)

    def forward(self, features: torch.Tensor, l1: int, l2: int, l3: int):
        """
        features are already in the right shape here (just a vector with m indices)
        """
        output = torch.zeros((features.shape[0], features.shape[1], 2*l1+1, 2*l2+1))
        cg = self.clebsch_gordan(l1, l2, l3)

        for m1 in np.arange(2 * l1 + 1):
            for m2 in np.arange(2 * l2 + 1):
                output[..., m1, m2] = torch.sum(cg[m1, m2, :] * features[..., :], axis=-1)
        
        return output


class SO3net(nn.Module):
    """
    A simple SO3-equivariant representation using spherical harmonics and
    Clebsch-Gordon tensor products.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_radial_basis: int,
        n_interaction_layers: int,
        use_residual_connections: bool = True,
        lmax: int = 2,
        cutoff: int = 5.0,
        z_max: int = 6,
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
        self.n_radial_basis = n_radial_basis
        self.n_interaction_layers = n_interaction_layers
        self.use_residual_connections = use_residual_connections
        # nonlinearity_type = "gate"

        self.lmax = lmax
        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(lmax=lmax)

        self.cutoff = cutoff
        self.z_max = z_max
        
        self.device = torch.device('cuda')

        # 1. embedding
        self.embedding = Embedding(lmax=lmax, z_max=z_max, n_atom_basis=n_atom_basis,
                                   cutoff=cutoff, n_radial_basis=n_radial_basis).to(self.device)

        # 2. interaction blocks using equivariant convolutions
        self.interaction_blocks = []

        x_input_irreps = Irreps(f"{self.n_atom_basis}x0e")
        for _ in range(self.n_interaction_layers):
            x_output_irreps = determine_feature_output_irreps(lmax, x_input_irreps, self.sh_irreps)
            self.interaction_blocks.append(
                SO3InteractionBlock(
                    input_irreps=x_input_irreps,
                    output_irreps=x_output_irreps,
                    sh_irreps=self.sh_irreps,
                    use_residual_connections=use_residual_connections,
                    device=self.device
                )
            )
            x_input_irreps = x_output_irreps

        # 3. change current features -> right amount of channels for each AO block to predict
        fulvene_orbitals = [0,0,1] * 2 + [0] * 4
        self.orbitals_degree = torch.Tensor(fulvene_orbitals)
        
        S = torch.ones(1, 1)
        P = torch.tensor([[0, 1.0, 0], [0, 0, 1], [1, 0, 0]])
        orbs = np.array([S, P])
        self.M = e3nn.math.direct_sum(*orbs[fulvene_orbitals].tolist()).to(self.device)
        
        # generate needed l3 combs for each orbital interactions
        self.orbitals_l3 = [[generate_l_combs(l1, l2) for l1 in self.orbitals_degree] for l2 in self.orbitals_degree]

        self.orb_feats_needed = [0, 0, 0]
        for orb_i in self.orbitals_l3:
            for orb_j in orb_i:
                for el in orb_j:
                    self.orb_feats_needed[el] += 1

        orbs_feats_irrep = Irreps(f"{self.orb_feats_needed[0]}x0e+{self.orb_feats_needed[1]}x1o+{self.orb_feats_needed[2]}x2e")
                
        # 4. hamiltonian prediction stuff
        self.interaction_blocks.append(
                SO3InteractionBlock(
                    input_irreps=x_input_irreps,
                    output_irreps=orbs_feats_irrep,
                    sh_irreps=self.sh_irreps,
                    use_residual_connections=use_residual_connections,
                    device=self.device
                )
            )
        self.tpe = TensorProductExpansion(device=self.device)
        
    def _generate_graph_edges(self, n_nodes: int, positions: torch.Tensor):
        full = torch.Tensor([[i for _ in range(n_nodes - 1)] for i in range(n_nodes - 1)]).to(self.device)
        idx = torch.triu_indices(*full.shape)
        edge_src = full[idx[0], idx[1]].type(torch.long)
        edge_dst = full.T[idx[0], idx[1]].type(torch.long) + 1
        edge_vec = positions[:, edge_dst] - positions[:, edge_src]
        return edge_src, edge_dst, edge_vec
        
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
        positions = inputs[properties.position].to(self.device)
        atomic_numbers = inputs[properties.Z].to(self.device)
        batch_size = atomic_numbers.shape[0]
        n_nodes = atomic_numbers.shape[-1]

        # make fully connected graph
        edge_src, edge_dst, edge_vec = self._generate_graph_edges(n_nodes, positions)

        # 1. embedding
        x, edge_radial, edge_sh = self.embedding(atomic_numbers, edge_vec)

        # 2. interaction blocks
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, edge_radial, edge_sh, edge_src, edge_dst, n_nodes)
    
        # 3. predict block wise hamiltonians
        size = int(torch.sum(2 * self.orbitals_degree + 1))
        H = torch.zeros((batch_size, n_nodes, size, size)).to(self.device)
        
        # list that keeps track of used features idxs for degree 0, 1, 2, etc...
        feature_index_counter = [0, 0, 0]

        curr_i = 0
        for idx_i, degree_i in enumerate(self.orbitals_degree):
            curr_j = 0
            for idx_j, degree_j in enumerate(self.orbitals_degree):
                degree_i = int(degree_i)
                degree_j = int(degree_j)

                # extract features from x corresponding to these orbital interactions
                current_features = []
                needed = self.orbitals_l3[idx_i][idx_j]
                for degree in needed:
                    if degree == 0:
                        start_idx = feature_index_counter[0]
                        end_idx = start_idx + 1
                    elif degree == 1:
                        start_idx = self.orb_feats_needed[0] + feature_index_counter[1] * 3
                        end_idx = start_idx + 3
                    elif degree == 2:
                        start_idx = self.orb_feats_needed[0] + self.orb_feats_needed[1] * 3 + feature_index_counter[2] * 5
                        end_idx = start_idx + 5
                    current_features.append(x[..., start_idx:end_idx])

                # fill hamiltonian block of orbital i with idx_i & degree_i <-> orbital j with idx_j & degree_j
                # by summing over the TPE's of allowed combinations of degree_i & degree_j
                H[..., curr_i:curr_i + (2*degree_i+1), curr_j:curr_j + (2*degree_j+1)] = torch.sum(torch.stack(
                    [self.tpe(current_features[idx], l1=degree_i, l2=degree_j, l3=l3) for idx, l3 in enumerate(generate_l_combs(degree_i, degree_j))]
                ), axis=0)

                curr_j += 2*degree_j+1
            curr_i += 2*degree_i+1

        # average over atomic contributions
        H = torch.mean(H, axis=1)

        # symmetrize hamiltonian
        H = H + torch.transpose(H, -1, -2)

        # transform from e3nn basis to original basis
        # H = self.M.T @ H @ self.M

        inputs['F'] = H.reshape(H.shape[0], -1)
        return inputs



        

