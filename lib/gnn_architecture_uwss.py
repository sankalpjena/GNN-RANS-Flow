"""
GNN Architecture for Underwater Simulation System

This module provides the Graph Neural Network (GNN) architecture used for the computational fluid dynamics simulation.

Third-party libraries for the project:

- PyTorch
- PyTorch-Geometric
- PyTorch-Lightning

User-defined libraries:

- lib.data_utils: Utility functions for data loading and normalization
- lib.log.logs: Logging utility for logging messages

Hyper-parameters:

- BATCH_SIZE_PER_GPU
- EDGE_CONV_MLP_LAYERS
- EDGE_CONV_MLP_SIZE
- DECODER_LAYER_SIZE_LIST
- USE_DECAY_LR
- FIXED_LEARNING_RATE
- INITIAL_LEARNING_RATE
- DECAY_FACTOR

Class Definitions:

1. GNNDataModule(pl.LightningDataModule):
   Data module for Graph Neural Network.

2. EdgeConvMLP(torch.nn.Module):
   Multi-Layer Perceptron (MLP) module for the GNN-EdgeConv.

3. AggregationMLP(torch.nn.Module):
   MLP module for local features aggregation.

4. LightningEdgeConvModel(pl.LightningModule):
   Lightning module for the GNN using EdgeConvolution.

    - training_step(self, train_batch, batch_idx) -> Tensor: Training step.
    - validation_step(self, val_batch, batch_idx) -> Tensor: Validation step.
    - test_step(self, test_batch, batch_idx) -> Tensor: Test step.
    - predict_step(self, pred_batch, batch_idx) -> Tensor: Prediction step.

    - configure_optimizers() -> list: Configures the optimizer and learning rate scheduler.

Author: Sankalp Jena
Date: 29 July 2023
"""

# Third-party libraries for the project

# PyTorch
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.nn import Sequential, Linear, ReLU, MSELoss, BatchNorm1d

# PyTorch-Geometric
import torch_geometric
from torch_geometric.nn import EdgeConv
from torch_geometric.loader import DataLoader

# PyTorch-Lightning
import pytorch_lightning as pl

# User-defined libraries
from lib import data_utils
from lib.log import logs 

# hyper-parameters
from hyper_params import BATCH_SIZE_PER_GPU,EDGE_CONV_MLP_LAYERS, EDGE_CONV_MLP_SIZE, DECODER_LAYER_SIZE_LIST, USE_DECAY_LR, FIXED_LEARNING_RATE, INITIAL_LEARNING_RATE, DECAY_FACTOR

num_cpus = 8  # Number of available CPUs, on Taurus NUMA core CPUs, == '--cpus-per-task=8'

# SEED
# Use a SEED_VALUE for reproducibility
SEED_VALUE = 2112

# Set the random seed value globally for torch_geometric
torch_geometric.seed_everything(seed=SEED_VALUE)

# Set the random seed value globally for pytorch_lightning
# sets seeds for numpy, torch, and python.random.
pl.seed_everything(SEED_VALUE, workers=True)

# input features
INPUT_FEATURES = 3 # [node_x_coord, node_y_coord, node_tag]
OUTPUT_LABELS = 3 # [ux, uy, wss]


class GNNDataModule(pl.LightningDataModule):
    """Data module for Graph Neural Network."""
    def __init__(self, pyg_graph_dict, noise_std=0):
        """
        Initialize the GNNDataModule.

        Args:
            pyg_graph_dict (dict): Dictionary containing PyTorch Geometric graph data.
            noise_std (float): Standard deviation of the Gaussian noise to be added.
        """
        super().__init__()
        self.pyg_graph_dict = pyg_graph_dict
        self.train_dataset_dict = None
        self.val_dataset_dict = None 
        self.test_dataset_dict = None
        self.noise_std = noise_std

    def setup(self, stage):
        """
        Setup the dataset splits and perform normalization.

        Args:
            stage (str): Stage of training (e.g., 'fit', 'validate', 'test').
        """
        self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict = data_utils.split_dataset(self.pyg_graph_dict)
        norm_stats = data_utils.get_minmax_normalization_stats(self.train_dataset_dict)
        self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict = data_utils.minmax_normalize_dataset(norm_stats, self.train_dataset_dict, self.val_dataset_dict, self.test_dataset_dict)

        if (stage == 'fit') and (self.noise_std!=0):
            # Add Gaussian noise to training data
            self.add_noise_to_dataset(self.train_dataset_dict)
    
    def add_noise_to_dataset(self, dataset_dict):
        """
        Add Gaussian noise to the dataset.

        Args:
            dataset_dict (dict): Dictionary containing the dataset to which noise is to be added.
        """
        for element in dataset_dict.values():
            # dict has [pyg_graph, Ub]
            # print("Before noise: ", element[0].y)
            element.y = element.y + torch.randn_like(element.y) * self.noise_std
            # print("After noise: ", element[0].y, '\n')
            # print('Shape: ', element[0].y.shape)
            # break
    
    def train_dataloader(self):
        """
        Return the data loader for the training dataset.

        Returns:
            DataLoader: Data loader for the training dataset.
        """
        if self.train_dataset_dict is None:
            raise RuntimeError("You must call `setup()` before accessing the dataloader.")
        train_dataset_list = [data for data in self.train_dataset_dict.values()]
        train_batch_size = BATCH_SIZE_PER_GPU
        train_loader = DataLoader(train_dataset_list, batch_size=train_batch_size, shuffle=True, num_workers=num_cpus,pin_memory=True)
        print(f'Train set divided in batches of {train_batch_size}')
        return train_loader

    def val_dataloader(self):
        """
        Return the data loader for the validation dataset.

        Returns:
            DataLoader: Data loader for the validation dataset.
        """
        if self.val_dataset_dict is None:
            raise RuntimeError("You must call `setup()` before accessing the dataloader.")
        val_dataset_list = [data for data in self.val_dataset_dict.values()]
        batch_size = BATCH_SIZE_PER_GPU
        val_loader = DataLoader(val_dataset_list, batch_size=batch_size, shuffle=False, num_workers=num_cpus,pin_memory=True)
        return val_loader

    def test_dataloader(self):
        """
        Return the data loader for the test dataset.

        Returns:
            DataLoader: Data loader for the test dataset.
        """
        if self.test_dataset_dict is None:
            raise RuntimeError("You must call `setup()` before accessing the dataloader.")
        test_dataset_list = [data for data in self.test_dataset_dict.values()]
        batch_size = BATCH_SIZE_PER_GPU
        test_loader = DataLoader(test_dataset_list, batch_size=batch_size, shuffle=False, num_workers=num_cpus,pin_memory=True)
        return test_loader

class EdgeConvMLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) module for the GNN-EdgeConv."""
    def __init__(self, in_channels=INPUT_FEATURES, out_channels=EDGE_CONV_MLP_SIZE, hidden_channels=EDGE_CONV_MLP_SIZE, num_layers=EDGE_CONV_MLP_LAYERS):
        """
        Initialize the MLP.

        Args:
            in_channels (int): Number of input channels = Node Features
            out_channels (int): Number of output channels = Embedded Space of the Node Features
            hidden_channels (int): Number of hidden channels = intermediate Embedded Space of the Node Features
            num_layers (int): Number of MLP layers, including the output layer
        """
        super().__init__()

        layers = []

        # hidden_layer_1
        layers.append(Linear(in_channels * 2, hidden_channels))
        layers.append(BatchNorm1d(hidden_channels))  # Add BatchNorm layer after Linear, and before Activation
        layers.append(ReLU())
        # can improve by adding nn.Dropout(drop_prob=0.4), after the Activation

        # Define rest hidden_layers
        for _ in range(num_layers - 2):
            layers.append(Linear(hidden_channels, hidden_channels))
            layers.append(BatchNorm1d(hidden_channels))  # Add BatchNorm layer
            layers.append(ReLU())

        # final output_layer without activation
        layers.append(Linear(hidden_channels, out_channels))
        layers.append(BatchNorm1d(out_channels))  # Add BatchNorm layer

        self.mlp = Sequential(*layers)

    def forward(self, x):
        """
        Perform forward pass through the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.mlp(x)

class AggregationMLP(torch.nn.Module):
    def __init__(self, in_channels=384, out_channels=1024):
        super(AggregationMLP, self).__init__()

        layers = []
        layers.append(Linear(in_channels, out_channels))
        layers.append(ReLU())
        # can improve by adding nn.Dropout(drop_prob=0.4), after the Activation

        self.mlp = Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class LightningEdgeConvModel(pl.LightningModule):
    """Lightning module for the GNN using EdgeConvolution."""
    def __init__(self, node_features=INPUT_FEATURES, node_labels=OUTPUT_LABELS):
        """
        Initialize the LightningEdgeConvModel.

        Args:
            node_features (int): Number of node features.
            node_labels (int): Number of node labels.
        """
        super(LightningEdgeConvModel, self).__init__()

        # Hyper-parameters
        # EdgeConv layers
        self.L = 8 # Number of EdgeConv
        self.hidden_neurons=EDGE_CONV_MLP_SIZE # number of neurons in EdgeConvMLP
        self.decoder_layers=DECODER_LAYER_SIZE_LIST #[1024, 512, 256]
        
        # L = 10 EdgeConv layers
        # Define 10 convolutional layers
        self.conv1 = EdgeConv(nn=EdgeConvMLP(in_channels=node_features, out_channels=self.hidden_neurons))
        self.conv2 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        self.conv3 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        self.conv4 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        self.conv5 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        self.conv6 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        self.conv7 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        self.conv8 = EdgeConv(nn=EdgeConvMLP(in_channels=self.hidden_neurons, out_channels=self.hidden_neurons))
        
        # Local features (embedded space) aggregation
        maxpool_mlp_in = self.L * self.hidden_neurons
        self.maxpool_mlp = AggregationMLP(in_channels=maxpool_mlp_in, out_channels=self.decoder_layers[0])  # MLP for global features

        # Decoder features
        decoder_in = maxpool_mlp_in + self.decoder_layers[0] # LatentSpace
        
        # can improve by adding nn.Dropout(drop_prob=0.4), after the Activation
        # also can test with BatchNorm1d after Linear op
        self.decoder = Sequential(
            Linear(decoder_in, self.decoder_layers[0]),
            ReLU(),
            Linear(self.decoder_layers[0], self.decoder_layers[1]),
            ReLU(),
            Linear(self.decoder_layers[1], self.decoder_layers[2]),
            ReLU(),
            Linear(self.decoder_layers[2], node_labels)
        )

        self.loss = MSELoss()

    def forward(self, x, edge_index):
        """
        Perform forward pass through the GNN.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Graph connectivity.

        Returns:
            Tensor: Predicted output.
        """
        
        # Define 10 embeddings
        h1 = self.conv1(x, edge_index) # Size:[N_vx128]

        h2 = self.conv2(h1, edge_index) # Size:[N_vx128]

        h3 = self.conv3(h2, edge_index) # Size:[N_vx128]

        h4 = self.conv4(h3, edge_index) # Size:[N_vx128]

        h5 = self.conv5(h4, edge_index) # Size:[N_vx128]

        h6 = self.conv6(h5, edge_index) # Size:[N_vx128]

        h7 = self.conv7(h6, edge_index) # Size:[N_vx128]

        h8 = self.conv8(h7, edge_index) # Size:[N_vx128]

        # Concatenate EdgeConv's embedded spaces
        local_features = torch.cat((h1, h2, h3, h4, h5, h6, h7, h8), dim=1) # Size: [N_v x (h1+h2+h3+h4+h5+h6+h7+h8+h9+h10 = 1280)]

        # local_features = torch.cat((h1, h2, h3), dim=1) # Size: [N_v x (h1+h2+h3 = 384)]
        # print('\n')
        # print('local_features shape: ', local_features.shape)
     
        # Map the local_features [N_v x 640] to [N_v x 1024] to extract the global
        # features and then apply max pooling for 1D global descriptor
        
        # Pass the local features to aggregate and map to a size of [N_v x 1024]
        mapped_local_features = self.maxpool_mlp(local_features)
        # print('\n')
        # print('mapped_local_features shape: ', mapped_local_features.shape)
        # Max pool the mapped local features to form 1D global feature
        global_descriptor_1d = torch.max(mapped_local_features, dim=0, keepdim=True).values
        # print('\n')
        # print('global_descriptor_1d shape: ', global_descriptor_1d.shape)
     
        # Repeat global features to match the dimensions of local features
        global_features = global_descriptor_1d.repeat(local_features.size(0), 1)  # N_v x 1024
        # print('\n')
        # print('global_features shape: ', global_features.shape)
     
        # Concatenate global features with the concatenated local features
        global_local_concat_features = torch.cat((local_features, global_features), dim=1)  # N_v x 1664
        # print('\n')
        # print('global_local_concat_features shape: ', global_local_concat_features.shape)
     
        # Pass the concatenated features through the decoder
        output = self.decoder(global_local_concat_features)
        
        return output
    
    def training_step(self, train_batch, batch_idx):
        """
        Perform a training step.

        Args:
            train_batch (Batch): Batch of training data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Training loss.
        """
        out = self.forward(train_batch.x, train_batch.edge_index)
        loss = self.loss(out, train_batch.y)
        # Log the training loss to TensorBoard
        self.log('train_loss', loss, batch_size=train_batch.x.shape[0], sync_dist=True, prog_bar=True,on_step=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Perform a validation step.

        Args:
            val_batch (Batch): Batch of validation data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Validation loss.
        """
        out = self.forward(val_batch.x, val_batch.edge_index)
        loss = self.loss(out, val_batch.y)
        self.log('val_loss', loss, batch_size=val_batch.x.shape[0], sync_dist=True, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        logs.info(f"Test Loss: {loss.item():.6f}")
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Performs a test step in the training loop.

        Args:
            test_batch (torch.Tensor): Test batch containing input features.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the test step.
        """
        out = self.forward(test_batch.x, test_batch.edge_index)
        loss = self.loss(out, test_batch.y)
        self.log('test_loss', loss, batch_size=test_batch.x.shape[0], sync_dist=True, prog_bar=True)
        logs.info(f"Test Loss: {loss.item():.8f}")
        return loss


    def predict_step(self, pred_batch, batch_idx):
        """
        Performs a prediction step in the training loop.

        Args:
            pred_batch (torch.Tensor): Batch containing input features for prediction.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Predicted output.
        """
        out = self.forward(pred_batch.x, pred_batch.edge_index)
        return out


    def configure_optimizers(self):
        """
        This function is used to configure the optimizer for the model's parameters. It uses the Adam 
        optimizer, a method for efficient stochastic optimization that is well suited for problems that 
        are large in terms of data and/or parameters.
        
        The function can operate in two modes depending on the value of `USE_DECAY_LR`. If `USE_DECAY_LR` 
        is set to True, a learning rate scheduler is used that decays the learning rate over time according 
        to the formula provided in the lambda function. This is typically useful for problems where the 
        loss landscape changes over time or where we want the optimizer to make large updates in the early 
        stages of training and smaller updates later on.
        
        If `USE_DECAY_LR` is set to False, a fixed learning rate specified by `FIXED_LEARNING_RATE` is used 
        for the Adam optimizer. This is a simpler approach and can be effective for many problems, 
        particularly if the learning rate is tuned carefully.
        
        The function returns the optimizer, and the learning rate scheduler if `USE_DECAY_LR` is set to True.
        
        Returns:
            list: List containing the optimizer and potentially a learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=FIXED_LEARNING_RATE)
        
        if USE_DECAY_LR:
            # From Chen et. al. (2021)
            optimizer = torch.optim.Adam(self.parameters(), lr=INITIAL_LEARNING_RATE)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_initial / (1 + gamma * epoch))
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

