from Playground.playground_utils import PlaygroundUtils
from Utils.preprocessor_utils import PreprocessorUtils
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from pyntcloud import PyntCloud
import open3d as o3d
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import utm
import json
import alphashape


class LaneMarkerUtils:
    def __init__(self):
        pass

    @staticmethod
    def intensity_stats(intensity):
        intensity = intensity.flatten()
        min_intensity, max_intensity = np.min(intensity), np.max(intensity)
        mean_intensity = np.mean(intensity)

        std_dev = np.std(intensity)
        return min_intensity, max_intensity, mean_intensity, std_dev

    @staticmethod
    def intensity_filter(intensity, num_std):
        _, _, mean_intensity, std_dev_intensity = LaneMarkerUtils.intensity_stats(intensity)
        lower_bound = int(mean_intensity + std_dev_intensity)
        upper_bound = int(mean_intensity + (num_std * std_dev_intensity))
        bounds = [lower_bound, upper_bound]
        return bounds

    @staticmethod
    def plot_intensity_histogram(intensity, bounds, output_html):

        min_intensity, max_intensity, mean_intensity, std_dev = LaneMarkerUtils.intensity_stats(intensity)
        filter_lower_bound, filter_upper_bound = bounds
        # Calculate KDE
        kde = gaussian_kde(intensity)
        intensity_values = np.linspace(min_intensity, max_intensity, 1000)
        density = kde(intensity_values)

        # Create the histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=intensity,
            nbinsx=80,
            histnorm='probability density',
            name='Histogram',
            opacity=1
        ))

        # Add KDE line
        fig.add_trace(go.Scatter(
            x=intensity_values,
            y=density,
            mode='lines',
            name='KDE',
            line=dict(color='red')
        ))

        # Add mean and standard deviation lines
        fig.add_trace(go.Scatter(
            x=[mean_intensity, mean_intensity],
            y=[0, max(density)],
            mode='lines',
            name='Mean',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[mean_intensity + std_dev, mean_intensity + std_dev],
            y=[0, max(density)],
            mode='lines',
            name='+1 std',
            line=dict(color='green', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[mean_intensity - std_dev, mean_intensity - std_dev],
            y=[0, max(density)],
            mode='lines',
            name='-1 std',
            line=dict(color='green', dash='dash')
        ))
        # Add shaded region for intensity filter
        fig.add_trace(go.Scatter(
            x=[filter_lower_bound, filter_lower_bound, filter_upper_bound, filter_upper_bound],
            y=[0, max(density), max(density), 0],
            fill='toself',
            fillcolor='orange',
            opacity=0.3,
            line=dict(color='orange'),
            name='Intensity Filter'
        ))
        # Update layout
        fig.update_layout(
            title='Histogram, KDE, and Standard Deviation of Intensity',
            xaxis_title='Intensity',
            yaxis_title='Density / Frequency',
            xaxis=dict(tickmode='linear', dtick=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        # Save the plot as an HTML file
        fig.write_html(output_html)
        print(f"Histogram saved as {output_html}")

    @staticmethod
    def apply_intensity_filter(map_file, filter_range):
        lower, upper = filter_range
        map_cloud = PreprocessorUtils.read_point_cloud(map_file)
        map_df = map_cloud.points
        map_points = map_df.values

        intensity_axis = map_points[:, 3]
        intensity_mask = np.logical_and(intensity_axis >= lower, intensity_axis <= upper)

        inlier_points_df = map_df[intensity_mask]
        outlier_points_df = map_df[~intensity_mask]

        # Convert filtered DataFrames back to PyntClouds
        inlier_cloud = PyntCloud(inlier_points_df)
        outlier_cloud = PyntCloud(outlier_points_df)

        return inlier_cloud, outlier_cloud

    @staticmethod
    def apply_clustering(obj, eps, min_points, print_progress=True):
        cloud = PreprocessorUtils.read_point_cloud(obj)
        pcd = PlaygroundUtils.pyntcloud_to_open3d(obj)
        cloud_df = cloud.points

        # Apply DBSCAN clustering
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

        # Add labels to the DataFrame
        cloud_df['cluster'] = labels

        clusters = []
        noise_df = cloud_df[cloud_df['cluster'] == -1].copy()  # Noise points
        noise_df.drop(columns=['cluster'], inplace=True)
        noise_cloud = PyntCloud(noise_df)

        for label in np.unique(labels):
            if label != -1:  # Skip noise points
                mask = cloud_df['cluster'] == label
                cluster_df = cloud_df[mask].copy()
                cluster_df.drop(columns=['cluster'], inplace=True)
                clusters.append(cluster_df)

        # Concatenate all clusters into a single DataFrame
        clusters_df = pd.concat(clusters, ignore_index=True)
        clusters_cloud = PyntCloud(clusters_df)

        return clusters, clusters_cloud, noise_cloud

    @staticmethod
    def compute_hulls(clusters_list, alpha):
        hulls = []
        for cluster in tqdm(clusters_list, desc="Computing Hulls...", total=len(clusters_list)):
            cloud_obj = PyntCloud(cluster)
            cloud_points = cloud_obj.points.values
            xy_pts = cloud_points[:, :2]
            points = [Point(p[0], p[1]) for p in xy_pts]
            points_tuples = [(p.x, p.y) for p in points]
            try:
                alpha_shape = alphashape.alphashape(points_tuples, alpha)

                if alpha_shape.geom_type == 'Polygon':
                    boundary_points = list(alpha_shape.exterior.coords)
                    hulls.append(boundary_points)
                else:
                    for geom in alpha_shape.geoms:
                        boundary_points = list(geom.exterior.coords)
                        hulls.append(boundary_points)
            except Exception as e:
                print(f"Error computing alpha shape: {e}")
                continue

        return hulls

    @staticmethod
    def convert_hulls_to_latlon(hulls_list, offset):
        lat, lon = offset
        utm_x, utm_y, zn, zl = utm.from_latlon(lat, lon)

        latlon_hulls = []

        for hull in hulls_list:
            latlon = []
            for point in hull:
                x, y = point
                utm_local_x = x + utm_x
                utm_local_y = y + utm_y
                lat, lon = utm.to_latlon(utm_local_x, utm_local_y, zn, zl)
                latlon.append([lon, lat])
                latlon_matrix = np.array(latlon)
            latlon_hulls.append(latlon_matrix)

        return latlon_hulls

    @staticmethod
    def extract_geojson(latlon_hulls, output_file):
        features = []
        for i, hull in enumerate(latlon_hulls):
            closed_hull = np.concatenate((hull, [hull[0]]))
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [closed_hull.tolist()]
                },
                "properties": {"polygon": i + 1}
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_file, 'w') as file:
            json.dump(geojson, file)

        print(f"GeoJSON data written to {output_file}")

