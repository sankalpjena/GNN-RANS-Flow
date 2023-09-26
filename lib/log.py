# Source: CFL-MINDS: https://github.com/cfl-minds/gnn_laminar_flow/tree/main
# Adapted to create new log file with time-stamp

# Logging
import logging
from datetime import datetime

# Current timestamp
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Handler 1
stream = logging.StreamHandler()
stream_format = logging.Formatter("%(message)s")
stream.setFormatter(stream_format)
stream.setLevel(logging.INFO)

# Handler 2
file = logging.FileHandler(f"log_{timestamp}.log")
file.setLevel(logging.INFO)
file_format = logging.Formatter("%(asctime)s:%(message)s", datefmt="%m-%d %H:%M:%S  ")
file.setFormatter(file_format)

# Log
logs = logging.getLogger(__name__)
logs.setLevel(logging.DEBUG)
logs.addHandler(file)
logs.addHandler(stream)

