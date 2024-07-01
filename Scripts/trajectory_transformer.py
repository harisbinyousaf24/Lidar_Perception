from Utils.trajectory_transformation_utils import TrajectoryTransformerUtils
import yaml
import os
import numpy as np


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
        self.frame_index = params['TrajectoryTransformer']['frame_index']
        self.rot_offset = params['TrajectoryTransformer']['rot_offset']
        self.use_heading_from = params['TrajectoryTransformer']['use_heading_from']
        self.manual_heading = params['TrajectoryTransformer']['manual_heading']
        self.use_manual_georef = params['TrajectoryTransformer']['use_manual_georef']
        self.georef_start = params['TrajectoryTransformer']['georef_start']

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

        self.GPS_global = TrajectoryTransformerUtils.read_gps(self.gnss_file)
        self.GPS_local = TrajectoryTransformerUtils.convert_global_to_local(self.GPS_global)
        self.ODOM_poses, self.ODOM_translations = TrajectoryTransformerUtils.load_states(self.poses_file)

    def decide_georeferencing(self):
        if self.use_manual_georef:
            offset = TrajectoryTransformerUtils.compute_offset(self.georef_start)
            return offset
        else:
            latlon_gps = self.GPS_global[:, :2]
            offset = TrajectoryTransformerUtils.compute_offset(latlon_gps[0])
            return offset

    def decide_rotation(self):
        if self.use_heading_from == 'gps':
            angle = TrajectoryTransformerUtils.compute_angle(self.GPS_local, self.ODOM_translations, self.frame_index)
            net_angle = angle + self.rot_offset
            R4 = TrajectoryTransformerUtils.rot_matrix(net_angle)
            return R4
        elif self.use_heading_from == 'manual':
            angle = self.manual_heading
            net_angle = angle + self.rot_offset
            R4 = TrajectoryTransformerUtils.rot_matrix(net_angle)
            return R4

    def apply_transformation(self):
        TF_POSES = []
        TF_translations = []
        CONV_offset = self.decide_georeferencing()
        R4 = self.decide_rotation()
        POSES_copy = np.copy(self.ODOM_poses)
        # Applying Rotation
        for idx in range(len(POSES_copy)):
            pose = POSES_copy[idx]
            rotated_pose = np.dot(R4, pose)
            translation = rotated_pose[:3, 3]
            TF_POSES.append(rotated_pose)
            TF_translations.append(translation)

        TF_POSES = np.array(TF_POSES)
        TF_translations = np.array(TF_translations)

        # Applying Translation
        TF_global = TrajectoryTransformerUtils.convert_local_to_global(TF_translations, CONV_offset)
        ODOM_global = TrajectoryTransformerUtils.convert_local_to_global(self.ODOM_translations, CONV_offset)

        set1 = [("GPS", self.GPS_global[:, :2]), ("Lidar", ODOM_global)]
        set2 = [("GPS", self.GPS_global[:, :2]), ("Lidar", TF_global)]

        return TF_POSES, set1, set2, CONV_offset

    def write_results(self, TF_POSES, set1, set2):
        np.save(self.tf_poses_file, TF_POSES)
        print(f"Poses saved at {self.tf_poses_file}")
        TrajectoryTransformerUtils.plot(self.org_plot_file, set1)
        print(f"Raw plots saved at {self.org_plot_file}")
        TrajectoryTransformerUtils.plot(self.tf_plot_file, set2)
        print(f"Transformed plots saved at {self.tf_plot_file}")

    def run(self):
        TF_POSES, set1, set2, CONV_offset = self.apply_transformation()
        self.write_results(TF_POSES, set1, set2)
        return CONV_offset


