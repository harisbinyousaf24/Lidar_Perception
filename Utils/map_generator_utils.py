import numpy as np
from pyntcloud import PyntCloud
import os
from tqdm import tqdm
from Utils.preprocessor_utils import PreprocessorUtils
from Playground.playground_utils import PlaygroundUtils


class MapGenratorUtils:
    def __init__(self):
        pass

    @staticmethod
    def transform_frames(transformation_matrix, lidar_points):
        # Convert lidar points to homogeneous coordinates
        ones = np.ones((lidar_points.shape[0], 1))
        homogeneous_points = np.hstack((lidar_points[:, :3], ones))

        # Apply the transformation matrix
        transformed_points = transformation_matrix @ homogeneous_points.T

        # Convert back to Cartesian coordinates
        transformed_points = transformed_points.T[:, :3]

        # Append intensity back to the transformed points
        if lidar_points.shape[1] == 4:
            transformed_points = np.hstack((transformed_points, lidar_points[:, 3].reshape(-1, 1)))

        return transformed_points

    @staticmethod
    def plane_segmentation_mask(obj, dist_thresh, ransac_n, num_iters):
        cloud = PreprocessorUtils.read_point_cloud(obj)
        pcd = PlaygroundUtils.pyntcloud_to_open3d(obj)
        cloud_df = cloud.points

        # Apply plane segmentation
        plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iters)

        # Create a mask for inliers
        inlier_mask = np.zeros(len(pcd.points), dtype=bool)
        inlier_mask[inliers] = True

        # Filtering inlier points based on the mask
        inlier_points_df = cloud_df[inlier_mask]
        outlier_points_df = cloud_df[~inlier_mask]

        # Convert filtered DataFrames back to PyntClouds
        inlier_cloud = PyntCloud(inlier_points_df)
        outlier_cloud = PyntCloud(outlier_points_df)

        return inlier_cloud, outlier_cloud

    @staticmethod
    def generate_map(tf_poses, lidar_frames_dir):
        frames = sorted(os.path.join(lidar_frames_dir, file_name) for file_name in os.listdir(lidar_frames_dir) if
                        file_name.endswith('.ply'))

        assert len(tf_poses) == len(
            frames), f"Mismatch between length of poses {len(tf_poses)} and length of lidar frames {len(frames)}"
        tf_frame = []
        for idx, file in tqdm(enumerate(frames), desc='Generating Map: ', total=len(frames)):
            file_path = frames[idx]
            cloud = PyntCloud.from_file(file_path)
            points = cloud.points.values

            tf_pose = tf_poses[idx]
            tf_points = MapGenratorUtils.transform_frames(tf_pose, points)
            tf_frame.append(tf_points)

        tf_map = np.vstack(tf_frame)

        return tf_map

    @staticmethod
    def generate_ground_map(tf_poses, lidar_frames_dir, dist_thresh, ransac_n, num_iters):
        frames = sorted(os.path.join(lidar_frames_dir, file_name) for file_name in os.listdir(lidar_frames_dir) if
                        file_name.endswith('.ply'))

        assert len(tf_poses) == len(
            frames), f"Mismatch between length of poses {len(tf_poses)} and length of lidar frames {len(frames)}"

        tf_ground = []
        for idx, file in tqdm(enumerate(frames), desc='Generating Ground Map: ', total=len(frames)):
            file_path = frames[idx]
            cloud = PyntCloud.from_file(file_path)
            ground_cloud, _ = MapGenratorUtils.plane_segmentation_mask(cloud, dist_thresh, ransac_n, num_iters)
            ground_points = ground_cloud.points.values
            tf_pose = tf_poses[idx]
            tf_ground_points = MapGenratorUtils.transform_frames(tf_pose, ground_points)
            tf_ground.append(tf_ground_points)

        tf_ground = np.vstack(tf_ground)

        return tf_ground
