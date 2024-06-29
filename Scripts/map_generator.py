import yaml
import os
from Utils.trajectory_transformation_utils import TrajectoryTransformerUtils
from Utils.map_generator_utils import MapGenratorUtils
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pyntcloud import PyntCloud
import numpy as np


class MapGenerator:
    def __init__(self, output_dir, preprocessor):
        self.main_dir = output_dir
        self.preprocessor_flag = preprocessor

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
        self.colored_map = params["MapGenerator"]["colored_map"]
        self.ground_map = params["MapGenerator"]["ground_map"]
        self.distance_threshold = params["MapGenerator"]["distance_threshold"]
        self.ransac_n = params["MapGenerator"]["ransac_n"]
        self.num_iters = params["MapGenerator"]["num_iters"]

        # Set input paths
        if self.preprocessor_flag:
            self.frames_dir = os.path.join(self.main_dir, setts['Preprocessor']['frames_dir'])
        else:
            self.frames_dir = os.path.join(self.main_dir, setts['Extractor']['frames_dir'])

        self.poses_file = os.path.join(self.main_dir, setts['TrajectoryTransformer']['tf_poses_file'])

        # Set output paths
        self.module_dir = os.path.join(self.main_dir, setts['MapGenerator']['map'])
        os.makedirs(self.module_dir, exist_ok=True)
        self.map_file = os.path.join(self.main_dir, setts["MapGenerator"]['map_file'])
        if self.colored_map:
            self.colored_map_file = os.path.join(self.main_dir, setts['MapGenerator']['colored_map_file'])
        if self.ground_map:
            self.ground_map_file = os.path.join(self.main_dir, setts["MapGenerator"]['ground_map_file'])

    def run(self):
        poses, translations = TrajectoryTransformerUtils.load_states(self.poses_file)
        transformed_map = MapGenratorUtils.generate_map(poses, self.frames_dir)

        cloud = PyntCloud(pd.DataFrame(transformed_map, columns=['x', 'y', 'z', 'intensity']))
        cloud.to_file(self.map_file)
        print(f"Map file saved to {self.map_file}")

        if self.ground_map:
            ground_map = MapGenratorUtils.generate_ground_map(poses, self.frames_dir,
                                                              self.distance_threshold,
                                                              self.ransac_n, self.num_iters)

            ground_cloud = PyntCloud(pd.DataFrame(ground_map, columns=['x', 'y', 'z', 'intensity']))
            ground_cloud.to_file(self.ground_map_file)
            print(f"Ground Map file saved to {self.ground_map_file}")

        if self.colored_map:
            z_values = transformed_map[:, 2]

            norm = Normalize(vmin=np.min(z_values),
                             vmax=np.max(z_values))

            colors = plt.cm.nipy_spectral(norm(z_values))[:, :3]
            data = np.hstack((transformed_map[:, :3], colors))
            columns = ["x", "y", "z", "red", "green", "blue"]

            # Create a DataFrame and a new PyntCloud for the colored map
            colored_cloud = PyntCloud(pd.DataFrame(data, columns=columns))

            colored_cloud.to_file(self.colored_map_file)
            print(f"Colored Map file saved to {self.colored_map_file}")
