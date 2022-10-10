import logging
from typing import Callable
import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import torch
import schnetpack as schnetpack
# from schnetpack.data.datamodule import AtomsDataModule
import os

from model.loss_functions import mean_squared_error, symm_matrix_mse
from model.caschnet_model import create_orbital_model

def train_model(
    save_path: str,
    property: str = 'F',
    loss_fn: Callable = symm_matrix_mse,
    batch_size: int = 16,
    lr: float = 5e-4,
    epochs: int = 100,
    basis_set_size: int = 36,
    database_path: str = None,
    split_file: str = None,
    use_wandb: bool = False,
    create_model_fn = create_orbital_model,
    initial_model_path: str = None,
    cutoff: float = 5.0
  ):
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

  """ Initializing a dataset """
  dataset = schnetpack.data.datamodule.AtomsDataModule(
    datapath=database_path,
    batch_size=batch_size,
    split_file=split_file,
    transforms=[
      schnetpack.transform.ASENeighborList(cutoff=cutoff),
      schnetpack.transform.CastTo32()
    ],
    property_units={property: 1.0},
    num_workers=8,
    pin_memory=True,
    load_properties=[property],
  )

  """ Initiating the Model """
  model = create_model_fn(loss_function=loss_fn, lr=lr, output_property_key=property, basis_set_size=basis_set_size, cutoff=cutoff)

  if initial_model_path is not None:
    state_dict = torch.load(initial_model_path).state_dict()
    for key in list(state_dict.keys()):
      state_dict['model.' + key] = state_dict.pop(key)
    model.load_state_dict(state_dict)

  """ Just for testing purposes """
  # dataset.setup()
  # for idx, sample in enumerate(dataset.train_dataloader()):
  #   # output = model(sample)
  #   loss = model.training_step(sample, 0)
  #   # loss = model.training_step(sample, 1)
  #   print(loss)
  #   break


  # callbacks for PyTroch Lightning Trainer
  logging.info("Setup trainer")
  callbacks = [
      schnetpack.train.ModelCheckpoint(
          monitor="val_loss",
          mode="min",
          save_top_k=1,
          save_last=True,
          dirpath="checkpoints",
          filename="{epoch:02d}",
          # inference_path=save_path,
          model_path=save_path
      ),
      pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="epoch"),
  ]

  if use_wandb:
    # wandb_project = os.environ['WANDB_PROJECT']
    wandb_project = 'caschnet_pyscf'
    logger = WandbLogger(project=wandb_project)
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                        logger=logger,
                                        default_root_dir='./test/',
                                        max_epochs=epochs,
                                        accelerator='gpu',
                                        devices=1)
  else:
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                    default_root_dir='./test/',
                                    max_epochs=epochs,
                                    accelerator='gpu',
                                    devices=1)
  logging.info("Start training")
  trainer.fit(model, datamodule=dataset)