import logging
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import numpy as np
from .utilities import *
from typing import Tuple

# Ensure logging is configured to use the same log file
logger = logging.getLogger(__name__)


def remove_ground(points,points_roi, eps=0.4, min_samples=8, ransac_min_samples=100, z_offset=0.2, step_by_step_visualization=False, visualize=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:   
        """
        Estimate and remove ground from the given point cloud. If flags are set then visualize the process step-by-step and/or visualize the end result with the original scene.
        
        Parameters:
            eps: float
            min_samples: int
            ransac_min_samples: int
            z_offset: float
            step_by_step_visualization: boolean
            visualize: boolean
        
        Returns:
            - Ground points: nx3
            - Non-ground poins: nx3
            - points (ground + non-ground) in ROI with ground mask: nx4
        
        Raises:
            ValueError:
                - If the number of available ground points is smaller than the minimum required samples for RANSAC (ransac_min_samples).
        """
        original_point_cloud = points
        original_label = 0
        
        # Apply DBSCAN
        print("Applying DBSCAN clustering.")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(points_roi)
        print("DBSCAN clustering completed.")
       
        # Step 2: Estimate ground height using the 50th percentile of non-outlier points
        print("Estimating ground height.")
        non_outlier_points = points_roi[dbscan_labels != -1]
        ground_height = np.percentile(non_outlier_points[:, 2], 50)

        # Step 3: First round of RANSAC for ground plane fitting
        print("Performing first round of RANSAC for ground plane fitting.")
        ground_points = points_roi[points_roi[:, 2] < ground_height]
        ransac_min_samples = min(ransac_min_samples, len(ground_points))  # Adjust min_samples if needed
        ransac = RANSACRegressor(min_samples=ransac_min_samples, residual_threshold=z_offset)
        ransac.fit(ground_points[:, :2], ground_points[:, 2])
        print("First round of RANSAC completed.")
        
        # Step 4: Remove points above the fitted plane
        inlier_mask = ransac.inlier_mask_
        refined_ground_points = ground_points[inlier_mask]

        
        n_samples = len(ground_points)
        if n_samples < ransac_min_samples:
            raise ValueError(f"Number of available ground points ({n_samples}) is smaller than the minimum required samples for RANSAC ({ransac_min_samples}).")
        
        # Step 5: Second round of RANSAC for refined ground plane fitting
        print("Performing second round of RANSAC for refined ground plane fitting.")
        ransac.fit(refined_ground_points[:, :2], refined_ground_points[:, 2])
        plane_model = ransac.estimator_

        # Step 6: Identify final ground points after the second round of RANSAC
        ground_plane_z = plane_model.predict(points_roi[:, :2])
        ground_mask = points_roi[:, 2] <= (ground_plane_z + z_offset)
        print("Final ground points identified.")

        # Separate ground and non-ground points
        ground_points_final = points_roi[ground_mask]
        non_ground_points_final = points_roi[~ground_mask]
        ground_mask = ground_mask[:,np.newaxis]
        points_roi_with_ground_mask = np.hstack((points_roi,ground_mask))
        

        if visualize:
            # Create point clouds with distinct colors
            non_ground_pcd = create_point_cloud(non_ground_points_final, color=[0.7, 0.7, 0.7])  # Light gray for non-ground points
            ground_pcd = create_point_cloud(ground_points_final, color=[0, 1, 0])                # Green for ground points
            
            origial_pcd = create_point_cloud(original_point_cloud, color=[255, 0, 0])
            #ground_color_map = load_colour_map("./color_map_ground2.json")
            #colors = np.array([ground_color_map[point_label] for point_label in original_label])   #point label is the label for each point in the original point cloud (without ROI)
            #origial_pcd.colors = o3d.utility.Vector3dVector(colors)
            origial_pcd = origial_pcd.translate((0, 0, 20))

            # Combine the point clouds and visualize
            print("Visualizing final point clouds.")
            o3d.visualization.draw_geometries([non_ground_pcd, ground_pcd, origial_pcd])
        
        print("Ground removal process completed.")
        return ground_points_final, non_ground_points_final, points_roi_with_ground_mask

        
        
    
  
