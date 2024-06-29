import yaml
from trajectory_transformer import TrajectoryTransformer
from lidar_odometry import RunKissICP
from extractor import Extractor
from preprocessor import Preprocessor

if __name__ == '__main__':
    with open('Config/external_settings.yaml', 'r') as external_settings:
        try:
            ext_setts = yaml.safe_load(external_settings)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")

    bag_file = ext_setts['bag_file']
    output_dir = ext_setts['output_dir']

    dataExtractor = Extractor(bag_file, output_dir)
    dataExtractor.run()

    preProcessor = Preprocessor(output_dir)
    preProcessor.run()

    kissicp = RunKissICP(output_dir)
    kissicp.run()

    trajectoryTransformer = TrajectoryTransformer(output_dir)
    trajectoryTransformer.run()

