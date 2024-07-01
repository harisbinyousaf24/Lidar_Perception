import numpy as np
import json
import utm
import math
import plotly.graph_objects as go
import plotly.io as pio


class TrajectoryTransformerUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_states(file):
        data = np.load(file)
        translations = []
        for tf in data:
            translation = tf[:3, 3]
            translations.append(translation)

        return data, np.array(translations)

    @staticmethod
    def read_gps(file):
        with open(file, 'r') as json_file:
            data = json.load(json_file)

        lat = np.array(data['latitude'])
        lon = np.array(data['longitude'])
        alt = np.array(data['altitude'])

        data = np.column_stack((lat, lon, alt))
        return data

    @staticmethod
    def plot(filename1: str, traj: list):
        def lat_lng_bounds(latitudes, longitudes):
            min_lat, max_lat = min(latitudes), max(latitudes)
            min_lng, max_lng = min(longitudes), max(longitudes)
            return min_lat, min_lng, max_lat, max_lng

        def map_center_and_zoom(min_lat, min_lng, max_lat, max_lng):
            center_lat = (min_lat + max_lat) / 2
            center_lng = (min_lng + max_lng) / 2

            lat_diff = max_lat - min_lat
            lng_diff = max_lng - min_lng

            zoom_lat = math.log(180 / lat_diff) / math.log(2)
            zoom_lng = math.log(360 / lng_diff) / math.log(2)

            zoom = min(zoom_lat, zoom_lng)
            return center_lat, center_lng, zoom

        traces = []
        all_latitudes = []
        all_longitudes = []

        for t in traj:
            name, mat = t
            latitudes = mat[:, 0]
            longitudes = mat[:, 1]
            all_latitudes.extend(latitudes)
            all_longitudes.extend(longitudes)
            trace = go.Scattermapbox(
                lon=longitudes,
                lat=latitudes,
                mode='markers',
                marker=go.scattermapbox.Marker(size=8),
                name=name
            )
            traces.append(trace)

        min_lat, min_lng, max_lat, max_lng = lat_lng_bounds(all_latitudes, all_longitudes)
        center_lat, center_lng, zoom = map_center_and_zoom(min_lat, min_lng, max_lat, max_lng)

        fig = go.Figure(traces)

        fig.update_layout(
            mapbox=dict(
                # Replace with your Mapbox access token
                accesstoken='pk.eyJ1IjoiaGFyaXNiaW55b3VzYWYiLCJhIjoiY2xsMXpkcGNoMDlqZDNrcDZrbjdrNHZnNyJ9.nXLa2DYKp8vNfzxbJRokAQ',
                center=dict(lat=center_lat, lon=center_lng),
                zoom=zoom,
                style='satellite'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0.5)')
        )
        pio.write_html(fig, file=filename1)

    @staticmethod
    def compute_offset(georef):
        lat, lon = georef
        utm_x, utm_y, zone_num, zone_letter = utm.from_latlon(lat, lon)
        array = np.array([utm_x, utm_y])
        offset = [array, zone_num, zone_letter]
        return offset

    @staticmethod
    def convert_global_to_local(gps_data):
        latlon_gps = gps_data[:, :2]
        offset = TrajectoryTransformerUtils.compute_offset(latlon_gps[0])
        offset = offset[0]
        utm_mat = []
        for lat, lon in latlon_gps:
            ux, uy, _, _ = utm.from_latlon(lat, lon)
            utm_arr = np.array([ux, uy])
            utm_mat.append(utm_arr)

        utm_mat = np.array(utm_mat)
        local_mat = utm_mat - offset

        return local_mat

    @staticmethod
    def convert_local_to_global(local_data, offset):
        local_data = local_data[:, :2]
        utm_x, utm_y = offset[0]
        zone_num = offset[1]
        zone_letter = offset[2]
        global_mat = []
        for local_x, local_y in local_data:
            utm_X, utm_Y = local_x + utm_x, local_y + utm_y
            lat_x, lat_y = utm.to_latlon(utm_X, utm_Y, zone_num, zone_letter)
            lat_arr = np.array([lat_x, lat_y])
            global_mat.append(lat_arr)

        global_mat = np.array(global_mat)

        return global_mat

    @staticmethod
    def rot_matrix(heading_angle):
        angle_radians = np.radians(heading_angle)
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
                                    [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        return rotation_matrix

    @staticmethod
    def compute_angle(local_gps, odom_data, frame_idx):
        poses_data = odom_data[:, :2]
        gps_data = local_gps[:, :2]

        tangent_gps = gps_data[frame_idx] - gps_data[0]
        tangent_lidar = poses_data[frame_idx] - poses_data[0]

        tangent_gps_norm = tangent_gps / np.linalg.norm(tangent_gps)
        tangent_lidar_norm = tangent_lidar / np.linalg.norm(tangent_lidar)

        dot_product = np.dot(tangent_gps_norm, tangent_lidar_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product)
        sign = np.sign(tangent_gps[0] * tangent_lidar[1] - tangent_gps[1] * tangent_lidar[0])
        angle_radians = sign * angle_radians
        angle_degrees = np.degrees(angle_radians)
        angle_degrees = -1 * angle_degrees

        return angle_degrees
