import open3d as o3d
from Utils.preprocessor_utils import PreprocessorUtils


class PlaygroundUtils:
    def __init__(self):
        pass

    @staticmethod
    def pyntcloud_to_open3d(file_or_cloud):
        cloud = PreprocessorUtils.read_point_cloud(file_or_cloud)
        cloud_pts = cloud.points[['x', 'y', 'z', 'intensity']].values

        xyz = cloud_pts[:, :3]
        open3d_cloud = o3d.geometry.PointCloud()
        open3d_cloud.points = o3d.utility.Vector3dVector(xyz)
        return open3d_cloud

    @staticmethod
    def color_pcd(pcd, clr='r'):
        # cloud = PreprocessorUtils.read_point_cloud(file)
        # pcd = PlaygroundUtils.pyntcloud_to_open3d(cloud)

        if clr == 'r':
            color = [1, 0, 0]
            pcd.paint_uniform_color(color)
        elif clr == 'g':
            color = [0, 1, 0]
            pcd.paint_uniform_color(color)
        elif clr == 'b':
            color = [0, 0, 1]
            pcd.paint_uniform_color(color)

        return pcd

    @staticmethod
    def visualize_pcd(pcd):
        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def visualize_multiple_pcds(*pcds):
        clouds = [pcd for pcd in pcds]
        o3d.visualization.draw_geometries(clouds)
