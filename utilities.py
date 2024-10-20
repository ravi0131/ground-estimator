import json
import logging
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Tuple



def create_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)).astype(np.float64))
    return pcd


def filter_points_in_ROI(points: np.ndarray, x_range:Tuple[int,int] , y_range: Tuple[int,int]) -> np.ndarray:
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

    # Apply the conditions to filter the points within the ROI
    in_roi = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        
    # Use the conditions to index the original points array
    filtered_points = points[in_roi]
    return filtered_points