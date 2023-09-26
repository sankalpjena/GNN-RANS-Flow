"""
sanity_check.py

This script checks if the necessary hyperparameters are defined in the `hyper_params.py` file.
If any of the parameters are not defined, it logs an error message for the user to define them.
Finally, it logs the used hyperparameters for the user's reference.

Functions:
    - run_sanity_check: Executes the sanity check for the hyperparameters.
"""

# Import the required libraries
from lib.log import logs 

def run_sanity_check():
    """
    Executes the sanity check for the hyperparameters.
    This function imports the hyperparameters from `hyper_params.py` and checks if they are defined.
    If any of the hyperparameters are not defined, it logs an error message for the user to define them.
    Finally, it logs the used hyperparameters for the user's reference.
    """
    logs.info('Running sanity check for hyper-parameters ...\n')

    # Import the hyperparameters
    try:
        from hyper_params import EDGE_CONV_MLP_LAYERS, EDGE_CONV_MLP_SIZE, DECODER_LAYER_SIZE_LIST, NUM_GPUS, BATCH_SIZE, BATCH_SIZE_PER_GPU, MAX_EPOCHS, EARLY_STOPPING_PATIENCE, FIXED_LEARNING_RATE, USE_DECAY_LR, INITIAL_LEARNING_RATE, DECAY_FACTOR, TRAIN_NOISE_STD
    except ImportError:
        logs.info("ERROR: The hyper_params.py file does not exist. Please create this file and define the necessary variables.")

    # List of variable names
    variables = ['EDGE_CONV_MLP_LAYERS', 'EDGE_CONV_MLP_SIZE', 'DECODER_LAYER_SIZE_LIST', 'NUM_GPUS', 'BATCH_SIZE', 'BATCH_SIZE_PER_GPU', 'MAX_EPOCHS', 'EARLY_STOPPING_PATIENCE', 'FIXED_LEARNING_RATE', 'USE_DECAY_LR', 'INITIAL_LEARNING_RATE', 'DECAY_FACTOR', 'TRAIN_NOISE_STD']

    # Check if each variable is defined
    for variable in variables:
        try:
            value = eval(variable)
            logs.info(f"Used Hyperparameter - {variable}: {value}")
        except NameError:
            logs.info(f"ERROR: The variable {variable} is not defined in hyper_params.py. Please provide this value.")

    logs.info("Sanity check completed successfully.")
