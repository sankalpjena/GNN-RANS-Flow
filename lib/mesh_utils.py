"""
mesh_utils.py

This module provides utility functions for handling mesh data and creating graph representations from mesh files. It includes functions for reading mesh files, generating graphs, converting between different graph representations, and handling wall shear stress data.

Default Python libraries:
- typing: Type hinting support.
- os: Provides functions for interacting with the operating system.
- networkx: Library for creating, analyzing, and visualizing complex networks.
- re: Regular expression operations.

Third-party libraries for the project:
- NumPy: Library for numerical computing in Python.
- Pandas: Library for data manipulation and analysis.
- Matplotlib: Library for creating static, interactive, and animated visualizations.
- Meshio, VTK: Libraries for reading and writing mesh data.
- torch, torch_geometric: Libraries for deep learning with PyTorch and graph-based neural networks.

The module consists of the following functions:

1. get_mesh_graph: Generate a graph using the mesh file.
2. create_reverse_edges: Create reverse edges for an undirected graph in PyG.
3. nx_plot: Plots a NetworkX graph with node attributes.
4. pyg_to_nx: Converts a PyG graph to an undirected NetworkX graph with node positions and node attributes ('y') [ux, uy, wss].
5. pkl_to_pyg_wss: Generates a PyG graph for the Wall Shear Stress (WSS) model from mesh edges, nodes, and node output values.
6. pkl_to_pyg_uwss: Generates a PyG graph for the Velocity and Wall Shear Stress (UWSS) model from mesh edges, nodes, and node output values.
7. pkl_to_pyg_noUb: Generates a PyG graph for the Velocity and Wall Shear Stress without Ub model from mesh edges, nodes, and node output values.
8. get_number: Extracts the time value from the filename using a regular expression.
9. get_wss_df: Extracts the `wallShearStress` from the wall patches using vtkPolyDataReader().
10. wss_correction: Corrects the 'wss_np' array by modifying data at the specified corner points.
11. get_vtk_data: Reads the nodes/points and point_data (velocity, wall shear stress) from the OpenFOAM VTK data.
12. remove_common_values: Removes common values between 'wall' and 'fluid' keys in the dictionary.
13. apply_node_tag: Applies node tags to wall and fluid nodes.

For more detailed information about each function, please see the corresponding docstrings within the module.

Author: Sankalp Jena
Date: 29 July 2023
"""


# Default Python libraries
from typing import Union, List, Tuple, Dict
from os import PathLike
import networkx as nx
import re
# Third-party libraries for the project

# NumPy
import numpy as np

# Pandas
import pandas as pd

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt

# Meshio, VTK
import meshio
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# PyTorch
import torch

# PyTorch-Geometric
import torch_geometric
from torch_geometric.data import Data
# Set the random seed value globally
torch_geometric.seed_everything(seed=2112)

# Dictionary to map strings to numbers
# SU2 mesh-cell shapes have the following shape:integer encoding 
SU2_SHAPE_IDS = {
    'line': 3,
    'triangle': 5,
    'quad': 9,
}


def get_mesh_graph(mesh_filename: Union[str, PathLike], dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:
    """
    Source: https://github.com/locuslab/cfd-gcn
    
    Generate a graph using the mesh file.

    Args:
        mesh_filename (Union[str, PathLike]): The path to the mesh file.
        dtype (np.dtype, optional): Data type of the mesh points. Defaults to np.float32.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]: A tuple containing four elements: mesh nodes, mesh edges, mesh elements, and marker dictionary.
    """

    def get_rhs(s: str) -> str:
        """
        Extracts the right-hand side of a string.

        Args:
            s (str): The input string.

        Returns:
            str: The right-hand side of the input string.
        """
        return s.split('=')[-1]

    # Initialize a marker dictionary that stores the boundary elements in key:value format
    # MARKER_TAG:MARKER_ELEMENTS
    marker_dict = {}

    with open(mesh_filename) as f:
        for line in f:
            # For Node coordinates
            if line.startswith('NPOIN'):
                num_points = int(get_rhs(line))
                # Get the mesh points as a list of lists
                mesh_points = [[float(p) for p in f.readline().split()[:2]] for _ in range(num_points)]
                nodes = np.array(mesh_points, dtype=dtype)

            # For boundary elements
            if line.startswith('NMARK'):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith('MARKER_TAG')  # Check if the line starts with 'MARKER_TAG'
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    
                    # Read marker elements from the file and store them in a list
                    marker_elems = [[int(e) for e in f.readline().split()[-2:]] for _ in range(num_elems)]
                    
                    # Store marker_elems in marker_dict with marker_tag as the key
                    marker_dict[marker_tag] = marker_elems

            # For inner element connectivity
            if line.startswith('NELEM'):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))

                # Read all the edge lines
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS['triangle']:
                        n = 3  # Nodes of a triangle element
                        triangles.append(elem[1:1+n])
                    elif elem[0] == SU2_SHAPE_IDS['quad']:
                        n = 4  # Nodes of a quadrilateral element
                        quads.append(elem[1:1+n])
                    else:
                        raise NotImplementedError
                    
                    # Create edges out of the nodes
                    elem = elem[1:1+n]
                    edges += [[elem[i], elem[(i+1) % n]] for i in range(n)]
                
                edges = np.array(edges, dtype=np.int64).transpose()
                elems = [triangles, quads]

    return nodes, edges, elems, marker_dict



def create_reverse_edges(edges):
    """
    Create reverse edges for undirected graph in PyG.

    Mesh edges are of type [node1, node2]. This is considered as directed by PyG.
    To create an undirected graph in PyG, the reverse edges must exist. This function
    creates the reverse edges [node2, node1].

    Args:
        edges (torch.Tensor): Directed mesh edges.

    Returns:
        torch.Tensor: Undirected PyG graph edges.
    """
    # Extract the rows from the input tensor
    up_row = edges[0].numpy()
    down_row = edges[1].numpy()

    # Create reverse edges [node2, node1]
    reverse = np.array([down_row, up_row])

    # Convert the NumPy array to a PyTorch tensor
    reverse = torch.from_numpy(reverse)

    # Concatenate the original and reverse edges along the column (second) dimension
    edges = torch.cat([edges, reverse], dim=1)

    return edges

def nx_plot(nx_graph, cmap='RdBu'):
    """
    Plots a NetworkX graph with node attributes.

    Args:
        nx_graph (nx.Graph): NetworkX graph to plot.
        cmap (str): Optional. Name of the colormap. Default is 'RdBu'.

    Returns:
        None
    """
    # Get positions and node attribute values of nodes
    pos_dict = nx.get_node_attributes(nx_graph, 'pos')
    node_y = nx.get_node_attributes(nx_graph, 'y')

    # Extract node attribute values for each dimension
    node_ux = [values[0] for values in node_y.values()]  # ux
    node_uy = [values[1] for values in node_y.values()]  # uy
    node_wss = [values[2] for values in node_y.values()]  # wss

    # Plot x-velocity
    plt.figure()
    nx.draw(nx_graph, pos=pos_dict, with_labels=False, node_size=5, node_color=node_ux, cmap=cmap)
    # Add colorbar and legend
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(node_ux)
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('ux')
    plt.title('x-velocity')
    plt.xlabel('Lx')
    plt.ylabel('Ly')

    plt.figure()
    # Plot y-velocity
    nx.draw(nx_graph, pos=pos_dict, with_labels=False, node_size=5, node_color=node_uy, cmap=cmap)
    # Add colorbar and legend
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(node_uy)
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('uy')
    plt.title('y-velocity')
    plt.xlabel('Lx')
    plt.ylabel('Ly')

    plt.figure()
    # Plot wall shear stress
    nx.draw(nx_graph, pos=pos_dict, with_labels=False, node_size=5, node_color=node_wss, cmap=cmap)
    # Add colorbar and legend
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(node_wss)
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('wss')
    plt.title('Wall Shear Stress')
    plt.xlabel('Lx')
    plt.ylabel('Ly')

    # Show the plot
    plt.show()

def pyg_to_nx(pyg_graph):
    """
    Converts a PyG graph to an undirected NetworkX graph with node positions and node attributes ('y') [ux, uy, wss].

    Args:
        pyg_graph (Data): PyG Data object representing the graph.

    Returns:
        nx.Graph: NetworkX graph with node positions and attributes.
    """
    nx_graph = nx.Graph()

    # Add nodes to the NetworkX graph
    nx_graph.add_nodes_from(range(pyg_graph.num_nodes))

    # Add node positions to nodes
    for i, (x, y) in enumerate(pyg_graph.x[:, :2].tolist()):
        nx_graph.nodes[i]['pos'] = (x, y)

    # Add edges to the NetworkX graph
    nx_graph.add_edges_from(pyg_graph.edge_index.t().tolist())

    # Add 'y' attribute values to nodes
    for i, y_values in enumerate(pyg_graph.y.tolist()):
        nx_graph.nodes[i]['y'] = y_values

    return nx_graph


def pkl_to_pyg_wss(mesh_edges, mesh_nodes_coord, node_tags, Ub, U_np, wss_np):
    """
    Generates a PyG graph from mesh edges, nodes, and node output values.

    Args:
        mesh_edges (list): List of mesh edges.
        mesh_nodes_coord (ndarray): Array of mesh node coordinates.
        node_tags (dict): Dictionary containing node tags.
        Ub (float): Ub value.
        U_np (ndarray): Array of velocity values.
        wss_np (ndarray): Array of wall shear stress values.

    Returns:
        Data: PyG Data object representing the mesh graph for WSS model.
    """
    
    # Configure the features of graph nodes

    # Coordinate positions of the PyG nodes
    node_positions = torch.tensor(mesh_nodes_coord, dtype=torch.float32) 
    
    # Create a list to store the feature tensor values
    tensor_values = []

    # Iterate over the indices of node_positions
    for idx in range(len(node_positions)):
        if idx in node_tags['wall']:
            tensor_values.append([node_positions[idx, 0], node_positions[idx, 1], 0, Ub])
        elif idx in node_tags['fluid']:
            tensor_values.append([node_positions[idx, 0], node_positions[idx, 1], 1, Ub])

    # Feature matrix: [num_nodes, num_features_per_node]
    Fx = torch.tensor(tensor_values, dtype=torch.float32)

    # For the graph to be undirected, the edge_index has to be [2, mesh_edges+reverse_mesh_edges].
    # To achieve that, we reverse the edges and concatenate them with the existing ones.
    mesh_edges = torch.tensor(mesh_edges, dtype=torch.int64)  # or torch.long for integer values, needed by torch_scatter
    graph_edges = create_reverse_edges(mesh_edges)

    # Node labels: wall shear stress (tau_yx), needed for output
    node_output = torch.from_numpy(np.array([wss_np], dtype=np.float32))
    node_output = node_output.t()  # Transpose to have PyG.y as [num_nodes, node_output]
    
    # Create the PyG Data object
    mesh_graph = Data(x=Fx, edge_index=graph_edges, y=node_output)
    
    return mesh_graph

def pkl_to_pyg_uwss(mesh_edges, mesh_nodes_coord, node_tags, Ub, U_np, wss_np):
    """
    Generates a PyG graph from mesh edges, nodes, and node output values.

    Args:
        mesh_edges (list): List of mesh edges.
        mesh_nodes_coord (ndarray): Array of mesh node coordinates.
        node_tags (dict): Dictionary containing node tags.
        Ub (float): Ub value.
        U_np (ndarray): Array of velocity values.
        wss_np (ndarray): Array of wall shear stress values.

    Returns:
        Data: PyG Data object representing the mesh graph for UWSS model.
    """
    
    # Configure the features of graph nodes

    # Coordinate positions of the PyG nodes
    node_positions = torch.tensor(mesh_nodes_coord, dtype=torch.float32) 
    
    # Create a list to store the feature tensor values
    tensor_values = []

    # Iterate over the indices of node_positions
    for idx in range(len(node_positions)):
        if idx in node_tags['wall']:
            tensor_values.append([node_positions[idx, 0], node_positions[idx, 1], 0, Ub])
        elif idx in node_tags['fluid']:
            tensor_values.append([node_positions[idx, 0], node_positions[idx, 1], 1, Ub])

    # Feature matrix: [num_nodes, num_features_per_node]
    Fx = torch.tensor(tensor_values, dtype=torch.float32)

    # For the graph to be undirected, the edge_index has to be [2, mesh_edges+reverse_mesh_edges].
    # To achieve that, we reverse the edges and concatenate them with the existing ones.
    mesh_edges = torch.tensor(mesh_edges, dtype=torch.int64)  # or torch.long for integer values, needed by torch_scatter
    graph_edges = create_reverse_edges(mesh_edges)

    # Node labels: velocity (ux, uy), needed for output
    ux = U_np[:, 0]
    uy = U_np[:, 1]
    # Create a numpy array
    node_output = torch.from_numpy(np.array([ux, uy, wss_np], dtype=np.float32))
    node_output = node_output.t()  # Transpose to have PyG.y as [num_nodes, node_output]
    
    # Create the PyG Data object
    mesh_graph = Data(x=Fx, edge_index=graph_edges, y=node_output)
    
    return mesh_graph

def pkl_to_pyg_noUb(mesh_edges, mesh_nodes_coord, node_tags, U_np, wss_np):
    """
    Generates a PyG graph from mesh edges, nodes, and node output values.

    Args:
        mesh_edges (list): List of mesh edges.
        mesh_nodes_coord (ndarray): Array of mesh node coordinates.
        node_tags (dict): Dictionary containing node tags.

        U_np (ndarray): Array of velocity values.
        wss_np (ndarray): Array of wall shear stress values.

    Returns:
        Data: PyG Data object representing the mesh graph for UWSS model.
    """
    
    # Configure the features of graph nodes

    # Coordinate positions of the PyG nodes
    node_positions = torch.tensor(mesh_nodes_coord, dtype=torch.float32) 
    
    # Create a list to store the feature tensor values
    tensor_values = []

    # Iterate over the indices of node_positions
    for idx in range(len(node_positions)):
        if idx in node_tags['wall']:
            tensor_values.append([node_positions[idx, 0], node_positions[idx, 1], 0])
        elif idx in node_tags['fluid']:
            tensor_values.append([node_positions[idx, 0], node_positions[idx, 1], 1])

    # Feature matrix: [num_nodes, num_features_per_node]
    Fx = torch.tensor(tensor_values, dtype=torch.float32)

    # For the graph to be undirected, the edge_index has to be [2, mesh_edges+reverse_mesh_edges].
    # To achieve that, we reverse the edges and concatenate them with the existing ones.
    mesh_edges = torch.tensor(mesh_edges, dtype=torch.int64)  # or torch.long for integer values, needed by torch_scatter
    graph_edges = create_reverse_edges(mesh_edges)

    # Node labels: velocity (ux, uy), needed for output
    ux = U_np[:, 0]
    uy = U_np[:, 1]
    # Create a numpy array
    node_output = torch.from_numpy(np.array([ux, uy, wss_np], dtype=np.float32))
    node_output = node_output.t()  # Transpose to have PyG.y as [num_nodes, node_output]
    
    # Create the PyG Data object
    mesh_graph = Data(x=Fx, edge_index=graph_edges, y=node_output)
    
    return mesh_graph

def get_number(filename):
    # Extract the time value from the filename using a regular expression
    # filename should be of format: *_$time.vtk
    match = re.search(r'.*_(\d+)\.vtk', filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # Return -1 if no number is found
    
def get_wss_df(wall_vtk_path):
    """
    Using vtkPolyDataReader() extracts the `wallShearStress` from the wall patches 
    Input: /path/to/wall_$time.vtk
    Output: pandas.Dataframe(columns=['x', 'y', 'z', 'tau_yx', 'tau_yy', 'tau_yz'])
    """
    
#     #cells = #points+1, so (128 cells + 1) x 2(~ for cells in z-axis ) = 258 points
#     Note: a cell is 3D in OpenFOAM, so a cell contains the info on xy plane and xy plane at some z distance
    
    # Create a reader for the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(wall_vtk_path)
    reader.Update()

    # Get the output of the reader
    output = reader.GetOutput()
    
    # Get the wall shear stress array and convert it to a NumPy array
    wss_array = output.GetPointData().GetArray('wallShearStress')
    wss_np = vtk_to_numpy(wss_array)

    # Get the points corresponding to the wall shear stress array
    points = output.GetPoints()
    points_np = vtk_to_numpy(points.GetData())

    # Concatenate the points and wss arrays along the columns axis
    data = np.concatenate((points_np, wss_np), axis=1)

    # Create a pandas DataFrame from the concatenated array
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'tau_yx', 'tau_yy', 'tau_yz'])

    # keep data at z=0
    wss_df = df[df['z'] == 0]
    
    return wss_df

def wss_correction(wss_np, corner_list, vtk_file_path):
    """
    Corrects the 'wss_np' array by modifying data at the specified corner points.

    Args:
        wss_np (numpy.ndarray): The array containing wall shear stress data.
        corner_list (list): List of corner point indices where the wss extrapolation is wrong.
        vtk_file_path (str): Path to the VTK data file.

    Returns:
        None: The 'wss_np' array will be modified in-place.
    """
    # get wss data from vtk

    # using regex extract the folder path and the latest time
    pattern = r'_([0-9]+)\.'
    match = re.search(pattern, vtk_file_path)
    if match:
        latest_time = int(match.group(1))
    else:
        print("Number not found.")

    pattern = r'(.*?/VTK/)'
    match = re.search(pattern, vtk_file_path)
    if match:
        case_vtk_folder = match.group(1)
    else:
        print("Substring not found.")

    bottomWall_vtk = case_vtk_folder + 'bottomWall/bottomWall_' + str(latest_time) + '.vtk'
    topWall_vtk = case_vtk_folder + '/topWall/topWall_' + str(latest_time) + '.vtk'

    # create pd.df from the vtk files
    bottomWall_wss = get_wss_df(bottomWall_vtk)
    topWall_wss = get_wss_df(topWall_vtk)

    # Get the 'tau_yx' values from rows 0 and 6 of bottomWall_wss and rows 7 and 13 of topWall_wss
    tau_yx_values_bw = bottomWall_wss.loc[[corner_list[0], corner_list[1]], 'tau_yx'].values
    tau_yx_values_tw = topWall_wss.loc[[corner_list[2], corner_list[3]], 'tau_yx'].values

    # Define the new values you want to assign
    new_wss_values = [tau_yx_values_bw[0], tau_yx_values_bw[1], tau_yx_values_tw[0], tau_yx_values_tw[1]]

    # Replace values at the specified indices
    wss_np[corner_list] = new_wss_values


def get_vtk_data(vtk_file_path):
    """
    Reads the nodes/points and point_data (velocity, wall shear stress) from the OpenFOAM VTK data.

    Args:
        vtk_file_path (str): Path to the VTK data file.

    Returns:
        tuple: Tuple containing 2D node points, velocity, and wall shear stress.
    """
    # Read the VTK data using meshio
    mesh = meshio.read(vtk_file_path)
    points = mesh.points
    point_data = mesh.point_data

    # Create pandas DataFrame for pruning data from z=0
    points_df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    U_df = pd.DataFrame(point_data['U'], columns=['ux', 'uy', 'uz'])
    wss_df = pd.DataFrame(point_data['wallShearStress'], columns=['tau_yx', 'tau_yy', 'tau_yz'])
    simulation_df = pd.concat((points_df, U_df, wss_df), axis=1)

    # Convert data to little-endian byte order
    simulation_df = simulation_df.astype({'x': '<f4', 'y': '<f4', 'z': '<f4', 'ux': '<f4', 'uy': '<f4', 'uz': '<f4', 'tau_yx': '<f4', 'tau_yy': '<f4', 'tau_yz': '<f4'})
    
    # Keep data at z=0
    simulation_2d_df = simulation_df[simulation_df['z'] == 0.0]
    
    # Drop unnecessary columns
    simulation_2d_df = simulation_2d_df.drop(['z', 'uz', 'tau_yy', 'tau_yz'], axis=1)

    # Specify big-endian byte order and a data type of 32-bit float to avoid byte order errors with torch.tensor
    points_np = np.array([simulation_2d_df['x'].values, simulation_2d_df['y'].values]).T
    U_np = np.array([simulation_2d_df['ux'].values, simulation_2d_df['uy'].values]).T

    wss_np = np.array(simulation_2d_df['tau_yx'].values)

    # modify data at the four corner points where the wss extrapolation is wrong
    corner_point_list = [0, 6, 7, 13]

    wss_correction(wss_np=wss_np, corner_list=corner_point_list, vtk_file_path=vtk_file_path)

    return points_np, U_np, wss_np


def remove_common_values(dict_data):
    """
    Remove common values between 'wall' and 'fluid' keys in the dictionary.

    Args:
        dict_data (dict): Dictionary containing 'wall' and 'fluid' keys.

    Returns:
        dict: Dictionary with common values removed from 'fluid' key.
    """
    # Convert 'wall' and 'fluid' values to sets
    wall_values = set(dict_data['wall'])
    fluid_values = set(dict_data['fluid'])

    # Find unique fluid values that are not present in wall_values
    unique_fluid_values = fluid_values.difference(wall_values)

    # Update 'fluid' key in the dictionary with unique fluid values
    dict_data['fluid'] = list(unique_fluid_values)

    return dict_data


def apply_node_tag(nested_list):
    """
    Apply node tags to wall and fluid nodes.

    The wall nodes are tagged as 0, and the fluid nodes are tagged as 1.

    Args:
        nested_list (list): Nested list containing wall and fluid nodes.

    Returns:
        dict: Dictionary with 'wall' and 'fluid' keys containing tagged nodes.
    """
    wall_node_list = nested_list[0]
    fluid_node_list = nested_list[1]

    node_tag_dict = {}

    # Create a flattened list of wall nodes
    flattened_list = [item for sublist in wall_node_list for item in sublist]

    # Assign unique tags to wall nodes and store them in 'wall' key
    node_tag_dict['wall'] = list(set(flattened_list))

    # Create a flattened list of fluid nodes
    flattened_list = [item for sublist in fluid_node_list for item in sublist]

    # Assign unique tags to fluid nodes and store them in 'fluid' key
    node_tag_dict['fluid'] = list(set(flattened_list))

    # Remove common values between 'wall' and 'fluid' keys
    node_tag_dict = remove_common_values(node_tag_dict)

    return node_tag_dict
