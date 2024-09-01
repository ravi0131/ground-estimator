import json
import logging
import numpy as np
import open3d as o3d
from typing import Dict
import matplotlib.pyplot as plt

def load_colour_map(color_map_file: str) -> Dict[int, np.ndarray]:
    """
    Load the color map from a JSON file.

    Args:
    color_map_file (str): Path to the JSON file containing the color map.

    Returns:
    Dict[int, np.ndarray]: A dictionary mapping labels to RGB colors.
    """
    with open(color_map_file) as f:
        color_map = json.load(f)
    logging.info("Color map loaded")
    return {int(k): np.array(v, dtype=np.float32) / 255.0 for k, v in color_map.items()}
def load_points_from_binary(file_path)-> np.ndarray:
    """
     Read a binary file containing point cloud data.

    Args:
    file_path (str): Path to the binary file.
    Returns:
    np.ndarray: An array of point cloud data.
    """
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # We only need x, y, z coordinates

def load_label(file_path: str) -> np.ndarray:
    """
    Read a binary file containing semantic labels.

    Args:
    file_path (str): Path to the label file.

    Returns:
    np.ndarray: An array of semantic labels.
    """
    label = np.fromfile(file_path, dtype=np.uint32)
    sem_label = label & 0xFFFF  # Semantic label in lower half
    return sem_label

def create_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)).astype(np.float64))
    return pcd



import pandas as pd
def export_point_cloud_to_csv(points):
    df = pd.DataFrame(points)
    df.to_csv('points_data.csv', index=False, header=False)    


def check_if_point_in_point_cloud(point_to_check,points):
    exists = np.any(np.all(np.isclose(points, point_to_check, atol=1e-7), axis=1))
    print(f"Does origin exist? {exists}")
    

def find_point_closest_to_sensor(points):  
    # Each row represents a point (x, y, z).
    # Calculate the Euclidean distance from the origin for each point
    distances = np.linalg.norm(points, axis=1)

    # Find the index of the minimum distance
    min_index = np.argmin(distances)

    # The closest point to the scanner
    closest_point = points[min_index]

    print(f"The closest point to the scanner is: {closest_point}")
from typing import Tuple
def filter_points_in_ROI(points: np.ndarray,labels: np.ndarray, x_range:Tuple[int,int] , y_range: Tuple[int,int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out points that are not in the region of interest.
    
    Parameters:
        points: np.ndarray
        labels: np.ndarray
        x_range: Tuple[int, int]
        y_range: Tuple[int, int]
        
    Returns:
        [Points in region of interest, Labels of those points]
    """
    # Define the ROI boundaries
    x_min, x_max = x_range[0], x_range[1]   # Longitudinal (x-axis)
    y_min, y_max = y_range[0], y_range[1]   # Lateral (y-axis)


    #labels = labels[:, np.newaxis]  
    points_and_labels = np.hstack((points,labels))
    # Apply the conditions to filter the points within the ROI
    in_roi = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        
    # Use the conditions to index the original points array
    filtered_points_and_labels = points_and_labels[in_roi]
    
    # filtered_points now contains only the points within the specified ROI
    print(f"Number of points in ROI: {filtered_points_and_labels.shape[0]}")
    return filtered_points_and_labels[:,:3], filtered_points_and_labels[:,3]

def visualize_point_cloud(labels, points, color_map= None, dark_mode=False):
    if color_map == None:
        # Assign a color to each cluster label
        unique_labels = np.unique(labels)
        colors = plt.get_cmap("tab10")(labels / (unique_labels.max() if unique_labels.max() > 0 else 1))
    else:
        colors = np.array([color_map[label] for label in labels])        
    # Convert the points and colors to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add point cloud to the visualizer
    vis.add_geometry(pcd)
    
    # Set background color based on mode
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) if dark_mode else np.asarray([1, 1, 1])
    
    # Render the point cloud
    vis.run()
    
    # Cleanup
    vis.destroy_window()

