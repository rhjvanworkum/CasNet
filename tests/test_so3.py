from model.architecture.so3_representation import SO3net
import schnetpack.properties as properties
import torch


def test_so3_output_shape():
    inputs = {}
    inputs[properties.Z] = torch.Tensor([
        [6, 6, 1, 1, 1, 1]
    ]).type(torch.int32)
    inputs[properties.position] = torch.Tensor([[
        [-2.67011, 1.10705 , -0.01190],
        [-1.35108, 0.92958, 0.00060],
        [-3.29876, 0.46787,  -0.61683],
        [-3.12073, 1.89048, 0.58244],
        [-0.72052, 1.56121, 0.61454],
        [-0.89812, 0.15220,  -0.60303],
    ]]).type(torch.float32)
    
    so3_net = SO3net(
        n_atom_basis = 12,
        n_radial_basis = 12,
        n_interaction_layers = 2,
    )
    
    assert so3_net(inputs)['F'][0].shape[0] == 14 * 14
    
def test_so3_graph_edges():
    inputs = {}
    inputs[properties.Z] = torch.Tensor([
        [6, 6, 1, 1, 1, 1]
    ]).type(torch.int32)
    inputs[properties.position] = torch.Tensor([[
        [-2.67011, 1.10705 , -0.01190],
        [-1.35108, 0.92958, 0.00060],
        [-3.29876, 0.46787,  -0.61683],
        [-3.12073, 1.89048, 0.58244],
        [-0.72052, 1.56121, 0.61454],
        [-0.89812, 0.15220,  -0.60303],
    ]]).type(torch.float32)   
    
    so3_net = SO3net(
        n_atom_basis = 12,
        n_radial_basis = 12,
        n_interaction_layers = 2,
    )
    
    positions = inputs[properties.position]
    
    edge_src, edge_dst, edge_vec = so3_net._generate_graph_edges(6, positions)
    print(edge_src, edge_dst)
    
    
    
    
# test_so3_output_shape()
test_so3_graph_edges()