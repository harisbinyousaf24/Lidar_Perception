import os
import coloredlogs
import yaml
import logging
import time
from trajectory_transformer import TrajectoryTransformer
from lidar_odometry import RunKissICP
from extractor import Extractor
from preprocessor import Preprocessor


def setup_logging():
    # Define the log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Define custom log colors
    field_styles = {
        'asctime': {'color': 'green'},
        'levelname': {'color': 'yellow', 'bold': True},
        'message': {'color': 'red'}
    }

    # Install coloredlogs with the custom format and field styles
    coloredlogs.install(level='INFO', fmt=log_format, field_styles=field_styles)

    logger = logging.getLogger(__name__)
    return logger


if __name__ == '__main__':
    logging = setup_logging()

    with open('../Config/external_settings.yaml', 'r') as external_settings:
        try:
            ext_setts = yaml.safe_load(external_settings)
        except yaml.YAMLError as exc:
            logging.error(f"Error reading YAML file: {exc}")
            exit(1)

    bag_file = ext_setts['bag_file']
    output_dir = ext_setts['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        logging.info(f'Extractor module took {elapsed_time:.2f} seconds.\n')
    if run_preprocessor:
        start_time = time.time()
        logging.info('Running Preprocessor module...')
        preProcessor = Preprocessor(output_dir)
        preProcessor.run()
        elapsed_time = time.time() - start_time
        logging.info(f'Preprocessor module took {elapsed_time:.2f} seconds.\n')
    if run_kissICP:
        start_time = time.time()
        logging.info('Running RunKissICP module...')
        kissicp = RunKissICP(output_dir)
        kissicp.run()
        elapsed_time = time.time() - start_time
        logging.info(f'RunKissICP module took {elapsed_time:.2f} seconds.\n')
    if run_trajectoryTransformer:
        start_time = time.time()
        logging.info('Running TrajectoryTransformer module...')
        trajectoryTransformer = TrajectoryTransformer(output_dir)
        trajectoryTransformer.run()
        elapsed_time = time.time() - start_time
        logging.info(f'TrajectoryTransformer module took {elapsed_time:.2f} seconds.\n')
