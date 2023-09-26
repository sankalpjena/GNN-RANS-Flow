"""
Data Utilities for Graph Neural Network Training

This module provides utility functions for loading, processing, and normalizing data used in training a Graph Neural Network (GNN) for computational fluid dynamics simulation.

Functions:

1. load_pkl_data(pickled_files: list) -> tuple:
   Loads pickled simulation data and returns the loaded data, case numbers, Ub values, Reb values, AR values, and Hp values.

2. write_csv_file(case_numbers: list, obstacle_types: list, filename: str) -> None:
   Writes case numbers and obstacle types to a CSV file.

3. split_dataset(pyg_graph_dict: dict, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple:
   Splits the PyG graph dictionary into train, validation, and test datasets based on the provided ratios.

4. calculate_range(tensor: torch.Tensor) -> tuple:
   Calculates the minimum and maximum values along each dimension of a tensor.

5. print_dataset_range(dataset_dict: dict, name: str = 'default') -> None:
   Prints the range of values in the dataset.

6. get_minmax_normalization_stats(train_dataset_dict: dict) -> list:
   Calculates the minimum and maximum values for each feature dimension in the training dataset and exports the normalization statistics to a CSV file.

7. minmax_normalize_dataset(norm_stats: list, train_dataset_dict: dict, val_dataset_dict: dict, test_dataset_dict: dict) -> tuple:
   Normalizes the dataset using the provided normalization statistics.

8. read_minmax_normalization_stats(file_path: str = './data/minmax_normalization_stats.csv') -> list:
   Reads the normalization statistics from a CSV file.

9. inverse_minmax_normalize_prediction_graph(prediction_graph: torch_geometric.data.Data, norm_stats: list) -> torch_geometric.data.Data:
   Inverse normalizes the prediction graph using the provided normalization statistics.

10. minmax_normalize_input_graph(input_graph: torch_geometric.data.Data, norm_stats: list) -> torch_geometric.data.Data:
   Normalizes the input graph using the provided normalization statistics.

Author: Sankalp Jena
Date: 29 July 2023
"""


# Default libraries
import pickle
import csv

# Project libraries
import torch
import torch_geometric
from torch_geometric.data import Data

# User libraries
from lib.log import logs

# Set the random seed value globally
torch_geometric.seed_everything(seed=2112)

def load_pkl_data(pickled_files, obstacle_type='default'):
    """
    Loads pickled simulation data and returns the loaded data, case numbers, Ub values, Reb values, AR values, and Hp values.

    Args:
        pickled_files (list): List of file paths of pickled files.

    Returns:
        tuple: A tuple containing the pickled data (list), case numbers (list), Ub values (list), Reb values (list),
        AR/theta values (list), and Hp values (list).
    """
    pickled_data = []  # List to store the loaded pickled data
    case_num = []  # List to store the extracted case numbers
    ub_value = []  # List to store the extracted Ub values
    reb_value = []  # List to store the extracted Reb values
    ar_theta_value = []  # List to store the extracted AR values
    hp_value = []  # List to store the extracted Hp values

    # Iterate over the pickled files
    for file_path in pickled_files:
        with open(file_path, "rb") as f:
            pickled_data.append(pickle.load(f))

        # file_path is PosixPath, convert to string for manipulation
        file_path = str(file_path)

        # Extract case number from the file name
        case_number = file_path.split("case_")[1].split("_ub")[0]
        case_num.append(case_number)

        # Extract Ub value from the file name
        ub = file_path.split("_ub_")[1].split("_reb")[0]
        ub_value.append(ub)

        # Extract Reb value from the file name
        reb = file_path.split("_reb_")[1].split("_ar")[0]
        reb_value.append(reb)

        if obstacle_type=='triangle':
            # Extract theta value from the file name
            theta = file_path.split("_theta_")[1].split("_hp")[0]
            ar_theta_value.append(theta)
        else:
            # Extract AR value from the file name
            ar = file_path.split("_ar_")[1].split("_hp")[0]
            ar_theta_value.append(ar)

        # Extract Hp value from the file name
        hp = file_path.split("_hp_")[1].split(".pkl")[0]
        hp_value.append(hp)

    return pickled_data, case_num, ub_value, reb_value, ar_theta_value, hp_value

def write_csv_file(case_numbers, obstacle_types, filename):
	"""
	Writes case numbers and obstacle types to a CSV file.

	Args:
	    case_numbers (list): List of case numbers.
	    obstacle_types (list): List of obstacle types.
	    filename (str): Name of the CSV file to write.

	Returns:
	    None
	"""
	with open(filename, mode='w') as file:
	    writer = csv.writer(file)
	    writer.writerow(["obstacle_type", "case_number"])  # Write header row
	    for case_number, obstacle_type in zip(case_numbers, obstacle_types):
	        writer.writerow([obstacle_type, case_number])  # Write obstacle type and case number row



def split_dataset(pyg_graph_dict, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the PyG graph dictionary into train, validation, and test datasets based on the provided ratios.
    Also writes the case numbers, obstacle types, and Ub values of each dataset to separate CSV files.

    Args:
        pyg_graph_dict (dict): Dictionary containing PyG graphs as values and case numbers as keys.
        train_ratio (float, optional): Ratio of train dataset size. Defaults to 0.8.
        val_ratio (float, optional): Ratio of validation dataset size. Defaults to 0.1.

    Returns:
        tuple: Tuple containing train, validation, and test dataset dictionaries.
    """
    # Extract the case numbers and obstacle types from the dictionary keys
    case_numbers = [key[1] for key in pyg_graph_dict.keys()]
    obstacle_types = [key[0] for key in pyg_graph_dict.keys()]

    # Define the sizes for train, validation, and test sets
    total_size = len(case_numbers)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the case numbers and obstacle types into train, validation, and test sets
    train_case_numbers = case_numbers[:train_size]
    train_obstacle_types = obstacle_types[:train_size]
    val_case_numbers = case_numbers[train_size:train_size + val_size]
    val_obstacle_types = obstacle_types[train_size:train_size + val_size]
    test_case_numbers = case_numbers[train_size + val_size:]
    test_obstacle_types = obstacle_types[train_size + val_size:]

    # Write the case numbers, obstacle types, and Ub values to CSV files
    write_csv_file(train_case_numbers, train_obstacle_types, './data/train_dataset.csv')
    write_csv_file(val_case_numbers, val_obstacle_types, './data/val_dataset.csv')
    write_csv_file(test_case_numbers, test_obstacle_types, './data/test_dataset.csv')

    # Return the split dictionaries with obstacle_type included
    train_dataset_dict = {key: pyg_graph_dict[key] for key in zip(train_obstacle_types, train_case_numbers)}
    val_dataset_dict = {key: pyg_graph_dict[key] for key in zip(val_obstacle_types, val_case_numbers)}
    test_dataset_dict = {key: pyg_graph_dict[key] for key in zip(test_obstacle_types, test_case_numbers)}

    logs.info('Data set is split into %s training set, %s validation set, and %s test set!\n',
              train_size, val_size, test_size)

    return train_dataset_dict, val_dataset_dict, test_dataset_dict

def calculate_range(tensor):
    """
    Calculates the minimum and maximum values along each dimension of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        tuple: Tuple containing the minimum and maximum values along each dimension.
    """
    min_value = tensor.min(dim=0).values.numpy()
    max_value = tensor.max(dim=0).values.numpy()
    return min_value, max_value

def print_dataset_range(dataset_dict, name='default'):
    """
    Prints the range of values in the dataset.

    Args:
        dataset_dict (dict): Dictionary containing the dataset.
        name (str): Name of the dataset.

    Returns:
        None
    """
    x_dataset = torch.cat([element.x for element in dataset_dict.values()], dim=0)
    y_dataset = torch.cat([element.y for element in dataset_dict.values()], dim=0)

    min_value_x, max_value_x = calculate_range(x_dataset)
    min_value_y, max_value_y = calculate_range(y_dataset)

    logs.info('Dataset: %s', name)
    logs.info('(pos_x, pos_y, node_tag) : [%s, %s]x[%s, %s]x[%s, %s]!',
              min_value_x[0], max_value_x[0], min_value_x[1], max_value_x[1],
              min_value_x[2], max_value_x[2])
    # model with 3 outputs
    if (min_value_y.shape[0] == 3):
        logs.info('(ux, uy, wss) : [%s, %s]x[%s, %s]x[%s, %s]!\n',
                  min_value_y[0], max_value_y[0], min_value_y[1], max_value_y[1], min_value_y[2], max_value_y[2])
    
    # model with 1 outputs
    else: 
        logs.info('(tau_yx) : [%s, %s]!\n',
                  min_value_y[0], max_value_y[0])

def get_minmax_normalization_stats(train_dataset_dict):
    """
    Calculates the minimum and maximum values for each feature dimension in the training dataset
    and exports the normalization statistics to a CSV file.

    Args:
        train_dataset_dict (dict): Dictionary containing training datasets with PyG graphs.

    Returns:
        list: List containing the minimum and maximum values for each feature dimension.
    """
    # Calculate the minimum and maximum values of each feature dimension in Data.x
    x_train = [element.x for element in train_dataset_dict.values()]  # Extract Data.x from training dataset
    x_train = torch.cat(x_train, dim=0)  # Concatenate all Data.x tensors
    
    min_x = x_train.min(dim=0).values
    max_x = x_train.max(dim=0).values
    
    # Calculate the minimum and maximum values of each feature dimension in Data.y
    y_train = [element.y for element in train_dataset_dict.values()]  # Extract Data.y from training dataset
    y_train = torch.cat(y_train, dim=0)  # Concatenate all Data.y tensors
    min_y = y_train.min(dim=0).values
    max_y = y_train.max(dim=0).values
    
    # Export normalization statistics to a CSV file
    # For model with 3 outputs
    if len(min_y.tolist()) == 3:
        stats = {
            'min_x[pos_x,pos_y,node_tag]': min_x.tolist(),
            'max_x[pos_x,pos_y,node_tag]': max_x.tolist(),
            'min_y[ux, uy, tau_yx]': min_y.tolist(),
            'max_y[ux, uy, tau_yx]': max_y.tolist()
        }

    else:
        stats = {
            'min_x[pos_x,pos_y,node_tag]': min_x.tolist(),
            'max_x[pos_x,pos_y,node_tag]': max_x.tolist(),
            'min_y[ux, uy, tau_yx]': min_y.tolist(),
            'max_y[ux, uy, tau_yx]': max_y.tolist()
        }

    csv_file_path = './data/minmax_normalization_stats.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    
    norm_stats = [min_x, max_x, min_y, max_y]
    return norm_stats

def minmax_normalize_dataset(norm_stats, train_dataset_dict, val_dataset_dict, test_dataset_dict):
    """
    Normalizes the dataset using the provided normalization statistics.

    Args:
        norm_stats (list): List containing the normalization statistics.
        train_dataset_dict (dict): Dictionary containing the training dataset.
        val_dataset_dict (dict): Dictionary containing the validation dataset.
        test_dataset_dict (dict): Dictionary containing the test dataset.

    Returns:
        tuple: Tuple containing the normalized training, validation, and test datasets.
    """
    # Unpack the normalization statistics
    min_x, max_x, min_y, max_y = norm_stats
    
    logs.info('Normalization stats are:\n')
    logs.info('min_x[pos_x,pos_y,node_tag]: %s', min_x)
    logs.info('max_x[pos_x,pos_y,node_tag]: %s', max_x)
    logs.info('min_y[ux, uy, tau_yx]: %s', min_y)
    logs.info('max_y[ux, uy, tau_yx]: %s', max_y)

    # Range of values before normalization (Data.x)
    logs.info('BEFORE:\n')
    print_dataset_range(train_dataset_dict, name='Training Set')
    print_dataset_range(val_dataset_dict, name='Validation Set')
    print_dataset_range(test_dataset_dict, name='Test Set')

    # Normalize feature vector (Data.x)
    for element in train_dataset_dict.values():
        element.x = (element.x - min_x) / (max_x - min_x)

    for element in val_dataset_dict.values():
        element.x = (element.x - min_x) / (max_x - min_x)

    for element in test_dataset_dict.values():
        element.x = (element.x - min_x) / (max_x - min_x)

    # Normalize target vector (Data.y)
    for element in train_dataset_dict.values():
        element.y = (element.y - min_y) / (max_y - min_y)

    for element in val_dataset_dict.values():
        element.y = (element.y - min_y) / (max_y - min_y)

    for element in test_dataset_dict.values():
        element.y = (element.y - min_y) / (max_y - min_y)

    # Output the range of values after normalization
    logs.info('AFTER:\n')
    print_dataset_range(train_dataset_dict, name='Training Set')
    print_dataset_range(val_dataset_dict, name='Validation Set')
    print_dataset_range(test_dataset_dict, name='Test Set')

    logs.info('Dataset normalized using Training Set statistics in ./data/minmax_normalization_stats.csv!')

    return train_dataset_dict, val_dataset_dict, test_dataset_dict

def read_minmax_normalization_stats(file_path='./data/minmax_normalization_stats.csv'):
    """
    Reads the normalization statistics from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the normalization statistics.

    Returns:
        list: List containing the normalization statistics.
    """
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip the header row
        stats = next(csv_reader)  # Read the statistics row

        # Parse the statistics as lists of floats
        min_x = [float(value) for value in stats[0].strip('[]').split(',')]
        max_x = [float(value) for value in stats[1].strip('[]').split(',')]
        min_y = [float(value) for value in stats[2].strip('[]').split(',')]
        max_y = [float(value) for value in stats[3].strip('[]').split(',')]

        normalization_stats = [min_x, max_x, min_y, max_y]

    return normalization_stats

def inverse_minmax_normalize_prediction_graph(prediction_graph, norm_stats):
    """
    Inverse normalizes the prediction graph using the provided normalization statistics.

    Args:
        prediction_graph (torch_geometric.data.Data): Prediction graph to be inverse normalized.
        normalization_stats (list): List containing the normalization statistics.

    Returns:
        torch_geometric.data.Data: Inverse normalized prediction graph.
    """
    min_x, max_x, min_y, max_y = norm_stats

    # Inverse normalize x values
    x_min = torch.tensor(min_x, dtype=torch.float32)
    x_max = torch.tensor(max_x, dtype=torch.float32)
    prediction_graph.x = prediction_graph.x * (x_max - x_min) + x_min

    # Inverse normalize y values
    y_min = torch.tensor(min_y, dtype=torch.float32)
    y_max = torch.tensor(max_y, dtype=torch.float32)
    prediction_graph.y = prediction_graph.y * (y_max - y_min) + y_min

    return prediction_graph



def minmax_normalize_input_graph(input_graph, norm_stats):
    """
    Normalizes the input graph using the provided normalization statistics.

    Args:
        norm_stats (list): List containing the normalization statistics.
        input_graph (torch_geometric.data.Data): Input graph to be normalized.

    Returns:
        torch_geometric.data.Data: Normalized input graph.
    """
    # Unpack the normalization statistics
    min_x, max_x, min_y, max_y = norm_stats
    min_x = torch.tensor(min_x, dtype=torch.float32)
    max_x = torch.tensor(max_x, dtype=torch.float32)
    min_y = torch.tensor(min_y, dtype=torch.float32)
    max_y = torch.tensor(max_y, dtype=torch.float32)

    # Normalize feature vector (Data.x)
    X = (input_graph.x - min_x) / (max_x - min_x)

    # Normalize target vector (Data.y)
    Y = (input_graph.y - min_y) / (max_y - min_y)

    normalized_input_graph = Data(x=X, edge_index=input_graph.edge_index, y=Y)
    return normalized_input_graph