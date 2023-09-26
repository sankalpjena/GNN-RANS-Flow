"""
This script defines the hyperparameters and training settings for the Graph Neural Network (GNN) model used in computational fluid dynamics mesh simulations. 

Hyperparameters:
- EDGE_CONV_MLP_LAYERS: The number of layers in the Multi-Layer Perceptron (MLP) used in EdgeConvolution. 
- EDGE_CONV_MLP_SIZE: The number of neurons per layer in the MLP used in EdgeConvolution.
- DECODER_LAYER_SIZE_LIST: The number of neurons in each layer of the decoder.

Training Parameters:
- NUM_GPUS: The number of GPUs available for distributed data parallel (DDP) multi-node GPU strategy.
- BATCH_SIZE: The total batch size for training, which was experimentally validated up to 20 with 4 GPUs.
- BATCH_SIZE_PER_GPU: The calculated batch size per GPU, derived by dividing the total batch size by the number of GPUs.
- MAX_EPOCHS: The maximum number of epochs for training the model.
- EARLY_STOPPING_PATIENCE: The number of consecutive epochs without improvement in validation loss before training is stopped for early stopping.

Learning Rate Parameters:
- FIXED_LEARNING_RATE: The learning rate used for training if USE_DECAY_LR is set to False.
- USE_DECAY_LR: A Boolean flag to decide whether to use a decaying learning rate or a fixed learning rate.
- INITIAL_LEARNING_RATE: The initial learning rate used if USE_DECAY_LR is set to True.
- DECAY_FACTOR: The decay factor applied to the learning rate per epoch if USE_DECAY_LR is set to True.

All the above parameters are required for the proper configuration and training of the GNN model.
"""


# Define GNN model hyperparameters
EDGE_CONV_MLP_LAYERS = 3 # Number of layers in the MLP; Default = 3
EDGE_CONV_MLP_SIZE = 64  # Number of neuros per layer of EdgeConvolution;  Default = 128
DECODER_LAYER_SIZE_LIST = [1024,512,256] # Number of neurons in each layer of Decoder;  Default = [1024,512,256]

# GNN Training Parameters

# Batch size based on number of GPUs for DDP multi-node-GPU strategy
NUM_GPUS = 4  # Number of GPUs
BATCH_SIZE = 16  # 1500 set: 12
BATCH_SIZE_PER_GPU = BATCH_SIZE // NUM_GPUS

# Training step parameters
MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 80

# Learning rate parameters
FIXED_LEARNING_RATE = 1e-4 # Default = 1e-4

USE_DECAY_LR = False  # Set to True to use decaying learning rate, False to use fixed learning rate
INITIAL_LEARNING_RATE = 1e-3  # Initial learning rate for decay; Default = 2e-3
DECAY_FACTOR = 1e-3  # Decay factor for decay learning rate; Default = 2e-3

# Training noise
TRAIN_NOISE_STD = 0.02 # Concept from MESHGRAPHNETS