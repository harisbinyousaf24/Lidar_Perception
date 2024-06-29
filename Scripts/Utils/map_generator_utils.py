import numpy as np
from pyntcloud import PyntCloud
import os
from tqdm import tqdm


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
