import yaml
from Utils.lane_marker_utils import LaneMarkerUtils
import os
from Utils.preprocessor_utils import PreprocessorUtils


class LaneMarker:
    def __init__(self, output_dir, offset):
        self.main_dir = output_dir
        self.offset = offset

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
        params = params['LaneMarker']
        self.plot_histogram = params['plot_histogram']
        self.num_std_devs = params['num_std_devs']
        self.use_manual_param = params['use_manual_param']
        self.intensity_filter = params['intensity_filter']
        self.eps = params['eps']
        self.num_points = params['num_points']
        self.print_progress = params['print_progress']
        self.alpha = params['alpha']

        # Setting up input paths
        self.map_file_path = os.path.join(self.main_dir, setts['MapGenerator']['ground_map_file'])

        # Setting up output paths
        self.module_dir = os.path.join(self.main_dir, setts['LaneMarker']['lanemarker'])
        os.makedirs(self.module_dir, exist_ok=True)
        self.plot_file = os.path.join(self.main_dir, setts['LaneMarker']['plot_path'])
        self.geojson_file = os.path.join(self.main_dir, setts['LaneMarker']['geojson_path'])

    def run(self):
        cloud = PreprocessorUtils.read_point_cloud(self.map_file_path)
        points = cloud.points.values
        intensity = points[:, 3]
        if not self.use_manual_param:
            self.intensity_filter = LaneMarkerUtils.intensity_filter(intensity, self.num_std_devs)
            print(f"Using computed intensity bounds: {self.intensity_filter}")
        if self.plot_histogram:
            print(f"Rendering Intensity Histogram. This may take a while...")
            LaneMarkerUtils.plot_intensity_histogram(intensity,
                                                     self.intensity_filter,
                                                     self.plot_file)
        print("Applying intensity filter...")
        inlier_intensity, _ = LaneMarkerUtils.apply_intensity_filter(self.map_file_path,
                                                                     self.intensity_filter)
        print("Intensity based filtering completed")
        print("Applying DBSCAN clustering...")
        clusters_list, _, _ = LaneMarkerUtils.apply_clustering(inlier_intensity,
                                                               self.eps, self.num_points,
                                                               self.print_progress)
        hulls = LaneMarkerUtils.compute_hulls(clusters_list, self.alpha)
        print("Converting hulls to latlon")
        latlon_hulls = LaneMarkerUtils.convert_hulls_to_latlon(hulls, self.offset)
        LaneMarkerUtils.extract_geojson(latlon_hulls, self.geojson_file)



