import time
from Scripts.distance_computer import DistanceComputer
from Scripts.trajectory_transformer import TrajectoryTransformer
from Scripts.lidar_odometry import RunKissICP
from Scripts.extractor import Extractor
from Scripts.preprocessor import Preprocessor
from Scripts.map_generator import MapGenerator


class LaunchSequence:
    def __init__(self, bag_file, output_dir, logger,
                 run_dataExtractor, run_preprocessor,
                 run_kissICP, run_trajectoryTransformer,
                 run_mapGenerator):

        start_time = time.time()
        logger.info('Estimating Drive Distance...')
        dstComputer = DistanceComputer(bag_file)
        distance = dstComputer.compute_distance()
        print(f"\033[93mTotal GPS estimated drive distance is {distance} meters.\033[0m")
        elapsed_time = time.time() - start_time
        logger.info(f'DistanceComputer module took {elapsed_time:.2f} seconds.\n')

        if run_dataExtractor:
            start_time = time.time()
            logger.info('Running Extractor module...')
            dataExtractor = Extractor(bag_file, output_dir)
            dataExtractor.run()
            elapsed_time = time.time() - start_time
            logger.info(f'Extractor module took {elapsed_time:.2f} seconds.\n')
        if run_preprocessor:
            start_time = time.time()
            logger.info('Running Preprocessor module...')
            preProcessor = Preprocessor(output_dir, distance)
            preProcessor.run()
            elapsed_time = time.time() - start_time
            logger.info(f'Preprocessor module took {elapsed_time:.2f} seconds.\n')
        if run_kissICP:
            start_time = time.time()
            logger.info('Running RunKissICP module...')
            kissicp = RunKissICP(output_dir, run_preprocessor)
            kissicp.run()
            elapsed_time = time.time() - start_time
            logger.info(f'RunKissICP module took {elapsed_time:.2f} seconds.\n')
        if run_trajectoryTransformer:
            start_time = time.time()
            logger.info('Running TrajectoryTransformer module...')
            trajectoryTransformer = TrajectoryTransformer(output_dir)
            trajectoryTransformer.run()
            elapsed_time = time.time() - start_time
            logger.info(f'TrajectoryTransformer module took {elapsed_time:.2f} seconds.\n')
        if run_mapGenerator:
            start_time = time.time()
            logger.info('Running MapGenerator module...')
            mapGen = MapGenerator(output_dir, run_preprocessor)
            mapGen.run()
            elapsed_time = time.time() - start_time
            logger.info(f'MapGenerator module took {elapsed_time:.2f} seconds.\n')

