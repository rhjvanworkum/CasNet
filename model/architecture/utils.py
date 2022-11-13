import math
import torch
import torch.nn as nn

@torch.jit.ignore
def computeEdgeIndex(
    r_max: float = None,
):
    """
    Compute edge indices between nodes within r_max.
    If there has already been an edge_index in batch, map all edge features to be consistent with the new edge indices.
    Zero-pad the features for new edges.
    """

    pos = data[key]
    pos = torch.as_tensor(pos, dtype=torch.get_default_dtype())        
    
    # per graph fully connected
    edge_index_lst = []
    cnt = 0
    for n_nodes in data['_n_nodes']:
        n_nodes = n_nodes.item()
        edge_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.long)
        edge_matrix += torch.arange(cnt, cnt+n_nodes)
        edge_index = torch.stack([edge_matrix.permute(1, 0).reshape(-1), edge_matrix.reshape(-1)])
        edge_index_lst.append(edge_index)
        cnt += n_nodes
    edge_index = torch.cat(edge_index_lst, dim=1).to(pos.device)
    
    # filter edges according to distance
    distance = pos[edge_index[0]] - pos[edge_index[1]]
    distance = torch.linalg.norm(distance, dim=-1)
    mask =  distance < r_max
    
    # filter edges according to custom criteria
    if not criteria is None:
        mask = torch.logical_or(mask, criteria(data, edge_index))
    mask = torch.logical_and(mask, torch.logical_not(edge_index[0]==edge_index[1]))

    def computeEdgeMap(a, b):
        j = 0
        lst = []
        for i in range(a.shape[1]):
            while not (b[:, j] == a[:, i]).all():
                j += 1
            lst.append(j)
        return torch.tensor(lst)
      
    if 'edge_index' in data:
        edge_map = computeEdgeMap(data['edge_index'], edge_index)
        mask[edge_map] = True
        
    mask = mask.expand((2, -1))
    edge_index = edge_index[mask].reshape(2, -1)
    
    # map edge attributes to new edge indices
    if 'edge_index' in data:
        edge_map = computeEdgeMap(data['edge_index'], edge_index)
        for key in attrs:
            if attrs[key][0] == 'edge': 
                tmp = data[key]
                data[key] = torch.zeros(edge_index.shape[1], data[key].shape[1], dtype=data[key].dtype).to(pos.device)
                data[key][edge_map] = tmp
                    
    if '_node_segment' in data:
        n_edges = torch.bincount(data['_node_segment'][edge_index[0]]).view(-1, 1)
    else:
        n_edges = torch.ones((1,), dtype=torch.long) * edge_index.shape[1]

    attrs["_n_edges"] = ('graph', '1x0e')
    data["_n_edges"] = n_edges
    
    data = {}
    data["edge_index"] = edge_index.to(pos.device)
    
    return data, attrs


@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

class PolynomialCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6, cutoff=_poly_cutoff):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)
        self.cutoff = cutoff

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        return self.cutoff(x, self._factor, p=self.p)

class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, r_min=0, num_basis=8, trainable=True, one_over_r=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
            
        one_over_r:
            Set to true if the value should explode at x = 0, e.g. when x is the interatomic distance.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.prefactor = 2.0 / (self.r_max - self.r_min)
        self.one_over_r = one_over_r

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / (self.r_max - self.r_min))
        result = self.prefactor * numerator
        if self.one_over_r:
            result = result/x.unsqueeze(-1)
        return  result