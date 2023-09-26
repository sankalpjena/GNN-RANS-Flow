# Default Python libraries
import time
from pathlib import Path

# Third-party libraries for the project

# PyTorch
import torch
from torchinfo import summary

# PyTorch-Lightning
import pytorch_lightning as pl

# PyTorch-Geometric
import torch_geometric

# User-defined libraries
from lib import gnn_architecture_uwss
from lib.log import logs


# Use a SEED_VALUE for reproducibility
SEED_VALUE = 2112

# Set the random seed value globally for torch_geometric
torch_geometric.seed_everything(seed=SEED_VALUE)

# Set the random seed value globally for pytorch_lightning
# sets seeds for numpy, torch, and python.random.
pl.seed_everything(SEED_VALUE, workers=True)

def load_model_weights(gnn_model, checkpoint_path):    
    checkpoint = torch.load(checkpoint_path)
    gnn_model.load_state_dict(checkpoint["state_dict"])
    logs.info("Model weights loaded from %s"%(checkpoint_path))
    return gnn_model

# Specify the data directory
data_dir = Path("./data/")
data_dir.mkdir(exist_ok=True)

def main():
    """
    Main function to test the Graph Neural Network model for computational fluid dynamics mesh simulation.
    """
    # Record the start time
    start_time = time.time()

    # Load the trained model
    gnn_model = gnn_architecture_uwss.LightningEdgeConvModel()

    # Load the checkpoint model weights and save as pt file
    checkpoint_path = "./checkpoints/gnn_L10_ABT3000.pt-epoch=0526-step=00079050-train_loss=0.000409-val_loss=0.000029.ckpt"
    gnn_model = load_model_weights(gnn_model, checkpoint_path)

    # save as torch.pt file
    model_path = "./gnn_L10_ABT3000_final.pt"
    torch.save(gnn_model.state_dict(), model_path)

    # model summary
    summary(gnn_model)

    # disable randomness, dropout, etc...
    gnn_model.eval()

    # Load data
    pyg_data_path = Path("/beegfs/ws/1/saje570c-obstacle-flow/surrogate_model/pyg_dataset/")  # UWSS model
    dataset_name = 'airfoil_bump_triangle_2998.pt'
    pyg_graph_dict = torch.load(pyg_data_path.joinpath(dataset_name))

    # Load dataset
    gnn_dm = gnn_architecture_uwss.GNNDataModule(pyg_graph_dict=pyg_graph_dict)

    # Call the setup method to process the dataset
    # gnn_dm.setup(stage=None)

    # Trainer
    trainer = pl.Trainer()
    test_results = trainer.test(model=gnn_model, datamodule=gnn_dm)

    # Record the end time
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time

    # Print the time taken
    logs.info("Time taken for testing: {:.2f} seconds".format(time_taken))

if __name__ == '__main__':
    main()
