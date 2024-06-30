from launcher import LaunchSequence
import coloredlogs
import yaml
import os
import time
import logging


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

    with open('Config/external_settings.yaml', 'r') as external_settings:
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
    run_mapGenerator = ext_setts['run_mapGenerator']
    run_laneMarker = ext_setts['run_laneMarker']

    COLOR = "\033[95m"
    RESET = "\033[0m"

    start_time = time.time()
    print(f"{COLOR}--------------------------------------------------------------------{RESET}")
    logging.info('INITIATING LAUNCH SEQUENCE...')
    print(f"{COLOR}--------------------------------------------------------------------{RESET}\n")

    launch = LaunchSequence(
        bag_file, output_dir, logging,
        run_dataExtractor, run_preprocessor,
        run_kissICP, run_trajectoryTransformer,
        run_mapGenerator, run_laneMarker
    )

    elapsed_time = time.time() - start_time
    print(f"{COLOR}--------------------------------------------------------------------{RESET}")
    logging.info(f'TOTAL TIME TOOK {elapsed_time:.2f} SECONDS.')
    print(f"{COLOR}--------------------------------------------------------------------{RESET}")
