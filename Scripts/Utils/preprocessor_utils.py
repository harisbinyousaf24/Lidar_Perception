from pyntcloud import PyntCloud
import numpy as np
from scipy.spatial import cKDTree


class PreprocessorUtils:
    def __init__(self):
        pass

    @staticmethod
    def read_point_cloud(file_or_cloud):
        if isinstance(file_or_cloud, PyntCloud):
            return file_or_cloud
        elif isinstance(file_or_cloud, str):
            cloud = PyntCloud.from_file(file_or_cloud)
            return cloud
        else:
            raise ValueError("Input must be a file path (str) or a Point Cloud object.")

    @staticmethod
    def remove_nans(file):
        pcd = PreprocessorUtils.read_point_cloud(file)
        points_df = pcd.points
        points_df.dropna(subset=['x', 'y', 'z'], inplace=True)
        return PyntCloud(points_df)

    @staticmethod
    def remove_ego(file, pog_z=-2.9):
        pcd = PreprocessorUtils.read_point_cloud(file)
        points_df = pcd.points
        points = points_df.values
        # EGO Dimensions
        ego_length, ego_width, ego_height = 4.27, 2.02, 1.45

        lidar_from_front = ego_length * 0.62
        lidar_from_rear = ego_length - lidar_from_front
        lidar_from_right = ego_width / 2
        lidar_roof_clearance = pog_z + ego_height

        # Masking
        # points = np.asarray(pcd.points)
        x_mask = np.logical_or(points[:, 0] < -lidar_from_rear,
                               points[:, 0] > lidar_from_front)
        y_mask = np.logical_or(points[:, 1] < -lidar_from_right,
                               points[:, 1] > lidar_from_right)
        z_mask = np.logical_or(points[:, 2] < pog_z,
                               points[:, 2] > 0)

        ego_mask = np.logical_or.reduce((x_mask, y_mask, z_mask))

        # Filtering points using the mask and keeping the intensity
        inlier_points = points_df[ego_mask]
        outlier_points = points_df[~ego_mask]

        # Convert filtered DataFrames back to PyntClouds, including intensity data
        inlier_cloud = PyntCloud(inlier_points)
        outlier_cloud = PyntCloud(outlier_points)

        return inlier_cloud, outlier_cloud

    @staticmethod
    def statistical_outlier_removal(file, params):
        k_nbs, z_thresh = params
        cloud = PreprocessorUtils.read_point_cloud(file)
        # Extract only the spatial coordinates for KDTree calculations
        spatial_points = np.array(cloud.points[['x', 'y', 'z']])

        # Create a KDTree for efficient nearest neighbor search
        kdtree = cKDTree(spatial_points)
        distances, _ = kdtree.query(spatial_points, k=k_nbs + 1)

        # Compute mean and standard deviation of distances to k nearest neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)  # exclude the first distance (self)
        std_dev = np.std(mean_distances)
        mean_of_mean_distances = np.mean(mean_distances)

        # Filter points based on the mean distance within standard deviation threshold
        mask = mean_distances <= (mean_of_mean_distances + z_thresh * std_dev)
        outlier_mask = mean_distances > (mean_of_mean_distances + z_thresh * std_dev)

        # Use masks to separate inliers and outliers, including all columns
        inlier_points_df = cloud.points[mask]
        outlier_points_df = cloud.points[outlier_mask]

        inlier_cloud = PyntCloud(inlier_points_df)
        outlier_cloud = PyntCloud(outlier_points_df)

        return inlier_cloud, outlier_cloud

    @staticmethod
    def z_filter(file, z_range):
        z_min, z_max = z_range
        pcd = PreprocessorUtils.read_point_cloud(file)
        points_df = pcd.points
        points = points_df.values

        z_axis = points[:, 2]
        z_mask = np.logical_and(z_axis >= z_min, z_axis <= z_max)

        # Filtering inlier points based on the mask
        inlier_points_df = points_df[z_mask]
        outlier_points_df = points_df[~z_mask]

        # Convert filtered DataFrames back to PyntClouds
        inlier_cloud = PyntCloud(inlier_points_df)
        outlier_cloud = PyntCloud(outlier_points_df)

        return inlier_cloud, outlier_cloud



