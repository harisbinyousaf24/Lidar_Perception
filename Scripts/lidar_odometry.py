import os
import subprocess
import yaml


class RunKissICP:
    def __init__(self, output_dir):
        self.main_dir = output_dir

        with open('../Params/params.yaml', 'r') as parameters:
            try:
                params = yaml.safe_load(parameters)
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file: {exc}")

        with open('../Config/settings.yaml', 'r') as settings:
            try:
                setts = yaml.safe_load(settings)
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file: {exc}")

        # Load parameters
        self.max_range = str(params['LidarOdometry']['max_range'])
        self.deskew = params['LidarOdometry']['deskew']

        # Load input paths
        self.preprocessed_frames = os.path.join(self.main_dir, setts['Preprocessor']['frames_dir'])

        # Load output paths
        self.odometry = setts['LidarOdometry']['odometry']
        self.module_dir = os.path.join(self.main_dir, self.odometry)
        os.makedirs(self.module_dir, exist_ok=True)

    def run(self):
        cwd = os.getcwd()
        command = ["kiss_icp_pipeline", self.preprocessed_frames,
                   "--max_range", self.max_range]
        if self.deskew:
            command.append("--deskew")

        try:
            os.chdir(self.module_dir)
            subprocess.run(command)
            os.chdir(cwd)
        except Exception as e:
            print(f"Error Running kiss_icp_pipeline: {e}")
