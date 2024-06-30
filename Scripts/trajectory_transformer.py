from Utils.trajectory_transformation_utils import TrajectoryTransformerUtils
import yaml
import os
import numpy as np
import utm


class TrajectoryTransformer:
    def __init__(self, output_dir):
        self.main_dir = output_dir

        with open('Params/params.yaml', 'r') as parameters:
            try:
                params = yaml.safe_load(parameters)
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file: {exc}")

        with open('Config/settings.yaml', 'r') as settings:
            try:
                setts = yaml.safe_load(settings)
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file: {exc}")

        # Load parameters
        self.frames_index = params['TrajectoryTransformer']['frames_index']
        self.offset = params['TrajectoryTransformer']['offset']
        self.use_heading_from = params['TrajectoryTransformer']['use_heading_from']
        self.manual_heading = params['TrajectoryTransformer']['manual_heading']
        self.manual_translate = params['TrajectoryTransformer']['manual_translate']

        # Load output paths
        self.trajectory_transformed = setts['TrajectoryTransformer']['transformed_trajectory']
        self.org_plot_file = os.path.join(self.main_dir, setts['TrajectoryTransformer']['org_plot_file'])
        self.tf_plot_file = os.path.join(self.main_dir, setts['TrajectoryTransformer']['tf_plot_file'])
        self.tf_poses_file = os.path.join(self.main_dir, setts['TrajectoryTransformer']['tf_poses_file'])

        self.module_dir = os.path.join(self.main_dir, self.trajectory_transformed)
        os.makedirs(self.module_dir, exist_ok=True)

        # Load input paths
        self.gnss_file = os.path.join(self.main_dir, setts['Extractor']['gnss_file'])
        self.latest_dir = os.path.join(self.main_dir, setts['LidarOdometry']['latest_dir'])
        for file in os.listdir(self.latest_dir):
            if file.endswith('.npy'):
                self.poses_file = os.path.join(self.latest_dir, file)

        self.gps_data = TrajectoryTransformerUtils.read_gps(self.gnss_file)
        self.local_gps, self.conversion_offset = TrajectoryTransformerUtils.convert_global_to_local(self.gps_data)
        if self.manual_translate:
            self.latlon_offset = params['TrajectoryTransformer']['latlon_offset']
            utm_x, utm_y, _, _ = utm.from_latlon(self.latlon_offset[0], self.latlon_offset[1])
            new_array = np.array([utm_x, utm_y])
            self.conversion_offset[0] = new_array
        self.poses, self.translations = TrajectoryTransformerUtils.load_states(self.poses_file)

    def decide_rotation(self):
        if self.use_heading_from == 'gps':
            angle = TrajectoryTransformerUtils.compute_rotation_based_on_gps(self.local_gps,
                                                                             self.translations,
                                                                             self.frames_index,
                                                                             self.offset)
            print(f"Using heading from GPS: {angle} degrees")
            r_matrix = TrajectoryTransformerUtils.rot_matrix(angle)
            return r_matrix
        if self.use_heading_from == 'manual':
            print(f"Using heading manually {self.manual_heading} degrees")
            r_matrix = TrajectoryTransformerUtils.rot_matrix(self.manual_heading)
            return r_matrix

    def apply_transformation(self):
        transformed_poses = []
        transformed_translations = []
        r_matrix = self.decide_rotation()
        for pose in self.poses:
            rotated_pose = np.dot(r_matrix, pose)
            translation = rotated_pose[:3, 3]
            transformed_poses.append(rotated_pose)
            transformed_translations.append(translation)

        tf_poses = np.array(transformed_poses)
        tf_translations = np.array(transformed_translations)
        return tf_poses, tf_translations

    def run(self):
        tf_poses, tf_translations = self.apply_transformation()
        np.save(self.tf_poses_file, tf_poses)
        print(f"Transformed poses saved at: {self.tf_poses_file}")
        # Plotting Raw plot
        raw_global_odom = TrajectoryTransformerUtils.convert_local_to_global(self.translations, self.conversion_offset)
        raw = [('GPS', self.gps_data[:, :2]), ('Lidar', raw_global_odom)]
        TrajectoryTransformerUtils.mapbox_trajectories(raw, self.org_plot_file)
        print(f"Raw trajectories plot saved at: {self.org_plot_file}")
        # Plotting Transformed plot
        tf_global_odom = TrajectoryTransformerUtils.convert_local_to_global(tf_translations, self.conversion_offset)
        tf = [('GPS', self.gps_data[:, :2]), ('Lidar', tf_global_odom)]
        TrajectoryTransformerUtils.mapbox_trajectories(tf, self.tf_plot_file)
        print(f"Transformed trajectories plot saved at: {self.tf_plot_file}")
        return self.conversion_offset


