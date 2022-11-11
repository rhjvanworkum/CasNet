"""
Script for training NN on CAS orbitals
"""
from model.loss_functions import mean_squared_error
from model.training import train_model
from typing import Callable
import torch
import schnetpack as spk
from model.architecture.model_output import ModelOutput, Hamiltonian

def create_model(loss_function: Callable,
                         lr: float = 5e-4,
                         output_property_key: str = 'F',
                         basis_set_size: int = 36,
                         cutoff: float = 5.0):

    pairwise_distance = spk.atomistic.PairwiseDistances()
    representation = spk.representation.SO3net(
        n_atom_basis=64,
        n_interactions=5,
        lmax=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_module = Hamiltonian(
        output_key=output_property_key,
        n_in=representation.n_atom_basis,
        n_layers=2,
        n_out=basis_set_size**2
    )
    nnp = spk.model.NeuralNetworkPotential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_module],
    )

    output = ModelOutput(
        name=output_property_key,
        loss_fn=loss_function,
        loss_weight=1.0,
        basis_set_size=basis_set_size
    )

    # Putting it in the Atomistic Task framework
    task = spk.AtomisticTask(
        model=nnp,
        outputs=[output],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": lr},
        # scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        # scheduler_args={'threshold': 1e-6, 'patience': 5},
        scheduler_cls=NoamLR,
        scheduler_args={'warmup_steps': 20},
        scheduler_monitor='val_loss'
    )

    return task

if __name__ == "__main__":
    epochs = 100
    lr = 1e-3
    batch_size = 16
    cutoff = 5.0
    basis_set_size = 36
    use_wandb = False

    loss_fn = mean_squared_error
    model_name =  'so3_test'
    database_path = './data_storage/fulvene_s01_200.db'
    split_file = './data_storage/geom_scan_200.npz'



    dataset = spk.data.datamodule.AtomsDataModule(
        datapath=database_path,
        batch_size=batch_size,
        split_file=split_file,
        transforms=[
            spk.transform.ASENeighborList(cutoff=cutoff),
            spk.transform.CastTo32()
        ],
        property_units={'F': 1.0},
        num_workers=8,
        pin_memory=True,
        load_properties=['F'],
    )

    pairwise_distance = spk.atomistic.PairwiseDistances()
    representation = spk.representation.SO3net(
        n_atom_basis=16,
        n_interactions=2,
        lmax=2,
        radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff),
    )

    dataset.setup()
    for idx, sample in enumerate(dataset.train_dataloader()):
        # sample is dict with values of size [batch_size X n_atoms, n_feat]
        out = pairwise_distance(sample)
        out = representation(sample)
        print(out['scalar_representation'].shape)
        print(out['multipole_representation'].shape)
        break



#   train_model(save_path='./checkpoints/' + model_name + '.pt',
#                   property=property, 
#                   loss_fn=loss_fn, 
#                   batch_size=batch_size, 
#                   lr=lr, 
#                   epochs=epochs,
#                   basis_set_size=basis_set_size,
#                   database_path=database_path,
#                   create_model_fn=create_model,
#                   split_file=split_file,
#                   use_wandb=use_wandb,
#                   cutoff=cutoff) 