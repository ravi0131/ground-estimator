import logging
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import numpy as np
from .utilities import *
from typing import Tuple

# Ensure logging is configured to use the same log file
logger = logging.getLogger(__name__)

class DuplicatePointError(Exception):
    """
    Exception raised for errors when a point cloud has duplicate points.
    """
    pass

class GroundEstimator:

    def initialize_map(self, points: np.ndarray, labels: np.ndarray):
        """
        Initialize the map with the given points, labels, and ground labels. Pass the entire point cloud and not just ROI
        :param points: All points in a point cloud (Nx3)
        :param labels: Labels of the points (Nx1)
        """
        logger.info("Initializing map with point cloud and labels.")
        self.points = points
        self.labels = labels
        self.ground_labels = np.array([6, 7, 8, 9, 10, 11, 12, 34, 35, 37, 38, 39])  # check labels.txt
        
        # Check for duplicates
        df = pd.DataFrame(points)
        if df.duplicated().any():
            logger.error("Duplicate points found in the point cloud.")
            raise DuplicatePointError("Points must be unique")
        
        self.map_to_label = dict(zip(map(tuple, points), labels))
        self.roi_x_range = [0, 40]
        self.roi_y_range = [-20, 20]
        self.points_roi , self.labels_roi = filter_points_in_ROI(points,labels,self.roi_x_range,self.roi_y_range)
        #self.map_roi = dict(zip(map(tuple,self.points_roi), self.labels_roi))
        logger.info("Map initialized successfully.")
    
    
    #TODO: Remove once you don't need it in remove-ground.ipynb. 
    def check_error_rate(self, estimated_ground_points: np.ndarray, estimated_non_ground_points: np.ndarray):
        """
        Checks whether an estimated ground point is labelled as ground in the original dataset and returns the number of correct and incorrect estimates.
        Parameters:
            estimated_ground_points: np.ndarray [nx3]
        DEPRECATED
        """
        logger.info("Checking error rate for estimated ground points.")
        positive_counter = 0
        negative_counter = 0
        for gpoint in estimated_ground_points:
            if self.map_to_label[tuple(gpoint)] in self.ground_labels:
                positive_counter += 1
            else:
                negative_counter += 1
        
        logger.info(f"Error rate checked: {positive_counter} positive, {negative_counter} negative.")
        return positive_counter, negative_counter
    

    def get_metrics(self, roi_with_mask: np.ndarray) -> Dict[str, float]:
        logger.info("Calculating accuracy of estimated ground points")
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for point_in_roi in roi_with_mask:
            estimated_as_ground = point_in_roi[3]
            coordinates = tuple(point_in_roi[:3])
            point_roi_label = self.map_to_label[coordinates]
            if estimated_as_ground:
                if point_roi_label in self.ground_labels:
                    true_positives += 1
                if point_roi_label not in self.ground_labels:
                    false_positives += 1
            if not estimated_as_ground:
                if point_roi_label not in self.ground_labels:
                    true_negatives += 1
                if point_roi_label in self.ground_labels:
                    false_negatives += 1
        accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
        precision = (true_positives)/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        f1_score = (2*true_positives)/(2*true_positives + false_positives + false_negatives)

        return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }
    
        
    
    def remove_ground(self, eps=0.4, min_samples=8, ransac_min_samples=100, z_offset=0.2, step_by_step_visualization=False, visualize=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:   
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
            Ground points: nx3
            Non-ground poins: nx3
            points (ground + non-ground) in ROI with ground mask: nx4
          
        """
        logger.info("Starting ground removal process.")
        original_point_cloud = self.points
        original_label = self.labels
        points_roi = self.points_roi
        
        # Apply DBSCAN
        logger.info("Applying DBSCAN clustering.")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(points_roi)
        logger.info("DBSCAN clustering completed.")
        
        if step_by_step_visualization:
            visualize_point_cloud(dbscan_labels, points_roi)
        
        # Step 2: Estimate ground height using the 50th percentile of non-outlier points
        logger.info("Estimating ground height.")
        non_outlier_points = points_roi[dbscan_labels != -1]
        ground_height = np.percentile(non_outlier_points[:, 2], 50)

        # Step 3: First round of RANSAC for ground plane fitting
        logger.info("Performing first round of RANSAC for ground plane fitting.")
        ground_points = points_roi[points_roi[:, 2] < ground_height]
        ransac_min_samples = min(ransac_min_samples, len(ground_points))  # Adjust min_samples if needed
        ransac = RANSACRegressor(min_samples=ransac_min_samples, residual_threshold=z_offset)
        ransac.fit(ground_points[:, :2], ground_points[:, 2])
        logger.info("First round of RANSAC completed.")
        
        # Step 4: Remove points above the fitted plane
        inlier_mask = ransac.inlier_mask_
        refined_ground_points = ground_points[inlier_mask]
        
        if step_by_step_visualization:
            visualize_point_cloud(dbscan_labels, refined_ground_points)
        
        # Step 5: Second round of RANSAC for refined ground plane fitting
        logger.info("Performing second round of RANSAC for refined ground plane fitting.")
        ransac.fit(refined_ground_points[:, :2], refined_ground_points[:, 2])
        plane_model = ransac.estimator_

        # Step 6: Identify final ground points after the second round of RANSAC
        ground_plane_z = plane_model.predict(points_roi[:, :2])
        ground_mask = points_roi[:, 2] <= (ground_plane_z + z_offset)
        logger.info("Final ground points identified.")

        # Separate ground and non-ground points
        ground_points_final = points_roi[ground_mask]
        non_ground_points_final = points_roi[~ground_mask]
        ground_mask = ground_mask[:,np.newaxis]
        points_roi_with_ground_mask = np.hstack((points_roi,ground_mask))
        
        if step_by_step_visualization:
            visualize_point_cloud(dbscan_labels, ground_points_final)
            visualize_point_cloud(dbscan_labels, non_ground_points_final)

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
            logger.info("Visualizing final point clouds.")
            o3d.visualization.draw_geometries([non_ground_pcd, ground_pcd, origial_pcd])
        
        logger.info("Ground removal process completed.")
        return ground_points_final, non_ground_points_final, points_roi_with_ground_mask
