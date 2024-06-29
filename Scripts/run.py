import yaml
import logging
import time
from trajectory_transformer import TrajectoryTransformer
from lidar_odometry import RunKissICP
from extractor import Extractor
from preprocessor import Preprocessor


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    setup_logging()

    with open('../Config/external_settings.yaml', 'r') as external_settings:
        try:
            ext_setts = yaml.safe_load(external_settings)
        except yaml.YAMLError as exc:
            logging.error(f"Error reading YAML file: {exc}")
            exit(1)

    bag_file = ext_setts['bag_file']
    output_dir = ext_setts['output_dir']
    run_dataExtractor = ext_setts['run_dataExtractor']
    run_preprocessor = ext_setts['run_preprocessor']
    run_kissICP = ext_setts['run_kissICP']
    run_trajectoryTransformer = ext_setts['run_trajectoryTransformer']

    if run_dataExtractor:
        start_time = time.time()
        logging.info('Running Extractor module...')
        dataExtractor = Extractor(bag_file, output_dir)
        dataExtractor.run()
        elapsed_time = time.time() - start_time
        logging.info(f'Extractor module took {elapsed_time:.2f} seconds.')
    if run_preprocessor:
        start_time = time.time()
        logging.info('Running Preprocessor module...')
        preProcessor = Preprocessor(output_dir)
        preProcessor.run()
        elapsed_time = time.time() - start_time
        logging.info(f'Preprocessor module took {elapsed_time:.2f} seconds.')
    if run_kissICP:
        start_time = time.time()
        logging.info('Running RunKissICP module...')
        kissicp = RunKissICP(output_dir)
        kissicp.run()
        elapsed_time = time.time() - start_time
        logging.info(f'RunKissICP module took {elapsed_time:.2f} seconds.')
    if run_trajectoryTransformer:
        start_time = time.time()
        logging.info('Running TrajectoryTransformer module...')
        trajectoryTransformer = TrajectoryTransformer(output_dir)
        trajectoryTransformer.run()
        elapsed_time = time.time() - start_time
        logging.info(f'TrajectoryTransformer module took {elapsed_time:.2f} seconds.')
