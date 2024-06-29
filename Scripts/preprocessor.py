from Utils.preprocessor_utils import PreprocessorUtils
import yaml
import os
from tqdm import tqdm


class Preprocessor:
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
        self.basic_preprocessing = params['Preprocessor']['basic_preprocessing']
        self.advanced_preprocessing = params['Preprocessor']['advanced_preprocessing']
        if self.basic_preprocessing:
            self.road_value_z = params['Preprocessor']['road_value_z']
        if self.advanced_preprocessing:
            self.SOR = [params['Preprocessor']['k_nbs'], params['Preprocessor']['z_thresh']]
            self.z_filter = params['Preprocessor']['z_filter']

        # Load input paths
        self.input_frames = os.path.join(self.main_dir, setts['Extractor']['frames_dir'])

        # Load output paths
        self.preprocessed_dir = setts['Preprocessor']['preprocessed']
        self.module_dir = os.path.join(self.main_dir, self.preprocessed_dir)
        os.makedirs(self.module_dir, exist_ok=True)

        self.frames_dir = os.path.join(self.main_dir, setts['Preprocessor']['frames_dir'])
        os.makedirs(self.frames_dir, exist_ok=True)

    def apply_basic_preprocessing(self):
        sorted_frames = sorted(
            [os.path.join(self.input_frames, frame) for frame in os.listdir(self.input_frames) if frame.endswith(".ply")])
        for file in tqdm(sorted_frames, desc='Basic Preprocessing', total=len(sorted_frames)):
            name = os.path.splitext(os.path.basename(file))[0]
            processed_file = os.path.join(self.frames_dir, name + '.ply')
            clean_cloud = PreprocessorUtils.remove_nans(file)
            filtered_cloud, _ = PreprocessorUtils.remove_ego(clean_cloud, self.road_value_z)
            filtered_cloud.to_file(processed_file)

        print(f"Basic Preprocessing Completed! Frames dumped at {self.frames_dir}")

    def apply_advanced_preprocessing(self):
        sorted_frames = sorted(
            [os.path.join(self.frames_dir, frame) for frame in os.listdir(self.frames_dir) if frame.endswith(".ply")])

        for file in tqdm(sorted_frames, desc='Advanced Preprocessing', total=len(sorted_frames)):
            filtered_cloud, _ = PreprocessorUtils.statistical_outlier_removal(file, self.SOR)
            filtered_cloud, _ = PreprocessorUtils.z_filter(filtered_cloud, self.z_filter)

            filtered_cloud.to_file(file)

        print(f"Advanced Preprocessing Completed! Frames dumped at {self.frames_dir}")

    def run(self):
        if self.basic_preprocessing:
            self.apply_basic_preprocessing()
        if self.advanced_preprocessing:
            self.apply_advanced_preprocessing()
