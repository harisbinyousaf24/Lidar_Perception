import yaml
import os
from pathlib import Path
from pyntcloud import PyntCloud
from tqdm import tqdm
from rosbags.highlevel import AnyReader
import pandas as pd
import json
from Utils.ply_to_pcd_converter import PLYtoPCD
import numpy as np


class Extractor:
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
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
        self.save_as = params['Extractor']['save_as']
        self.gps_topic = params['Extractor']['gps_topic']
        self.lidar_topic = params['Extractor']['lidar_topic']
        self.topics_list = [self.gps_topic, self.lidar_topic]

        # Load output paths
        self.data_dir = setts['Extractor']['data_dir']
        self.module_dir = os.path.join(self.main_dir, self.data_dir)
        os.makedirs(self.module_dir, exist_ok=True)

        self.frames_dir = os.path.join(self.main_dir, setts['Extractor']['frames_dir'])
        self.gnss_file = os.path.join(self.main_dir, setts['Extractor']['gnss_file'])

    def run(self):
        gps_ts, latitudes, longitudes, altitudes = [], [], [], []
        with AnyReader([Path(self.bag_file)]) as reader:
            connections = [x for x in reader.connections if x.topic in self.topics_list]
            for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections),
                                                       desc='Processing Bag...'):
                topic_name = connection.topic

                if topic_name == '/rslidar_points':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    points_data = np.frombuffer(msg.data, dtype=np.float32).reshape(
                        (msg.height, msg.width, 4))  # Assuming x, y, z, intensity

                    pcd = points_data[:, :, :3]
                    intensities = points_data[:, :, 3]
                    flat_pcd = pcd.reshape(-1, 3)
                    flat_intensities = intensities.reshape(-1, 1)

                    # Save to PCD
                    df = pd.DataFrame(data=np.hstack((flat_pcd, flat_intensities)),
                                      columns=['x', 'y', 'z', 'intensity'])
                    cloud = PyntCloud(df)
                    ply_file = os.path.join(self.frames_dir, f"{timestamp}.ply")

                    if self.save_as == 'pcd':
                        pcd_file = os.path.join(self.frames_dir, f"{timestamp}.pcd")
                        PLYtoPCD.ply_to_pcd(ply_file, pcd_file)
                        os.remove(ply_file)

                    elif self.save_as == 'ply':
                        cloud.to_file(ply_file)

                elif topic_name == '/gnss':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    gps_ts.append(timestamp)
                    latitudes.append(msg.latitude)
                    longitudes.append(msg.longitude)
                    altitudes.append(msg.altitude)

                # GPS data to JSON
            gps_data = {
                "timestamps": gps_ts,
                "latitude": latitudes,
                "longitude": longitudes,
                "altitude": altitudes
            }

            with open(self.gnss_file, 'w') as f:
                json.dump(gps_data, f, indent=4)

            print(f"Lidar frames saved in directory: {self.frames_dir}")
            print(f"GPS data saved at: {self.gnss_file}")
            