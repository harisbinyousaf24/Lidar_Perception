from geopy.distance import geodesic
from rosbags.highlevel import AnyReader
from pathlib import Path
import yaml


class DistanceComputer:
    def __init__(self, bag_path):
        self.bag_file = bag_path

        with open('Params/params.yaml', 'r') as parameters:
            try:
                params = yaml.safe_load(parameters)
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file: {exc}")

        # Load parameters
        self.gnss_topic = params['Topics']['gps_topic']

    def compute_distance(self):
        total_distance = 0.0
        previous_coords = None

        with AnyReader([Path(self.bag_file)]) as reader:
            connections = [x for x in reader.connections if x.topic == self.gnss_topic]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                current_coords = (msg.latitude, msg.longitude)

                if previous_coords is not None:
                    distance = geodesic(previous_coords, current_coords).meters
                    total_distance += distance

                previous_coords = current_coords

        return total_distance


