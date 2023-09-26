"""
Graph Neural Network Training Script

This script trains a Graph Neural Network (GNN) model for computational fluid dynamics simulation.
The GNN is trained using PyTorch and PyTorch-Lightning, and it utilizes PyTorch-Geometric for graph data handling.

The training process includes loading the dataset, setting up the training parameters, defining the GNN model,
loading pre-trained weights for transfer learning (if available), configuring loggers and callbacks, and running
the training and testing processes. The trained model is then saved, and the total training time is logged.

Author: Sankalp Jena

Date: 29 July 2023

Usage: python train_model.py

Note: Please ensure that the required data, model, and hyperparameter files are available and correctly configured.

"""

import time
from pathlib import Path
import torch
torch.cuda.empty_cache()
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from lib import data_utils
from lib import mesh_utils
from lib.log import logs
from lib import gnn_architecture_uwss  # UWSS model
from hyper_params import NUM_GPUS, MAX_EPOCHS, EARLY_STOPPING_PATIENCE, TRAIN_NOISE_STD
from lib import sanity_check

# Specify the data directory
data_dir = Path("./data/")
data_dir.mkdir(exist_ok=True)

# Use a SEED_VALUE for reproducibility
SEED_VALUE = 2112
torch_geometric.seed_everything(seed=SEED_VALUE)  # Set the random seed value globally for torch_geometric
pl.seed_everything(SEED_VALUE, workers=True)  # Set the random seed value globally for pytorch_lightning

def load_model_weights(gnn_model, checkpoint_path):
    """
    Load model weights from a checkpoint file.

    Args:
        gnn_model (pytorch_lightning.LightningModule): The GNN model to load the weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        pytorch_lightning.LightningModule: The GNN model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path)
    gnn_model.load_state_dict(checkpoint["state_dict"])
    logs.info("Model weights loaded from %s" % checkpoint_path)
    return gnn_model

def main():
    """
    Main function to train the Graph Neural Network model for computational fluid dynamics mesh simulation.
    """

    # Record the start time
    start_time = time.time()

    logs.info(f"Number of available GPUs:{torch.cuda.device_count()}")

    # function call sets the float32 precision for matrix multiplications to 'high', which enables Tensor Core utilization and trades off precision for performance
    # only with A100 GPUs
    torch.set_float32_matmul_precision('high')

    # Load data
    pyg_data_path = Path("/beegfs/ws/1/saje570c-obstacle-flow/surrogate_model/pyg_dataset/")  # UWSS model
    dataset_name = 'airfoil_bump_triangle_2998.pt'
    pyg_graph_dict = torch.load(pyg_data_path.joinpath(dataset_name))

    # Load dataset, for training use noise_std
    # UWSS model
    gnn_dm = gnn_architecture_uwss.GNNDataModule(pyg_graph_dict=pyg_graph_dict, noise_std=TRAIN_NOISE_STD)

    # Setup training parameters
    gnn_model = gnn_architecture_uwss.LightningEdgeConvModel()

    # If a previous model, with same weights exists, use it for initializing weights => Transfer Learning
    # checkpoint_path = ''
    # gnn_model = load_model_weights(gnn_model, checkpoint_path)

    # Create TensorBoard and CSV loggers
    tensorboard_logger = TensorBoardLogger(save_dir='logs/', name='tensorboard_logs')
    csv_logger = CSVLogger(save_dir='logs/', name='csv_logs')

    # Checkpoint callback
    gnn_model_name = 'gnn_L10_ABT3000.pt'
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='%s-{epoch:04d}-{step:08d}-{train_loss:.6f}-{val_loss:.6f}' % (gnn_model_name),
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        save_weights_only=False
    )

    # Define EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Metric to monitor (validation loss/MSE)
        mode='min',  # 'min' if the metric should be minimized, 'max' if it should be maximized
        patience=EARLY_STOPPING_PATIENCE,  # Number of epochs with no improvement before stopping
        verbose=True  # Whether to print early stopping information
    )

    # If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`.
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=NUM_GPUS,
        num_nodes=1,
        strategy='ddp_find_unused_parameters_true',  # strategy='ddp',
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,  # 10
        logger=[tensorboard_logger, csv_logger],
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Run training
    trainer.fit(model=gnn_model, datamodule=gnn_dm)

    # Save the trained model
    torch.save(gnn_model.state_dict(), gnn_model_name)

    # Test the model after training
    trainer = pl.Trainer(devices=1, callbacks=[checkpoint_callback])
    trainer.test(model=gnn_model, datamodule=gnn_dm, ckpt_path="best")

    # Record the end time
    end_time = time.time()

    time_taken = end_time - start_time

    # Convert time taken to hours
    time_taken_hours = time_taken / 3600

    # Print the time taken in hours
    logs.info("Time taken for training: {:.2f} hours".format(time_taken_hours))


if __name__ == '__main__':
    main()
