from plyfile import PlyData
import numpy as np


class PLYtoPCD:
    def __init__(self):
        pass

    @staticmethod
    def ply_to_pcd(ply_file, pcd_file):
        # Read the PLY file data
        ply_data = PlyData.read(ply_file)
        vertex_data = ply_data['vertex'].data

        # Extract coordinate and intensity data
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        intensity = vertex_data['intensity']

        # Stack these arrays into a single numpy array
        data = np.stack((x, y, z, intensity), axis=-1)
        num_points = data.shape[0]

        # Prepare the PCD file header
        header = f"""# .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z intensity
        SIZE 4 4 4 4
        TYPE F F F F
        COUNT 1 1 1 1
        WIDTH {num_points}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {num_points}
        DATA ascii
        """

        # Write the header and data to the PCD file
        with open(pcd_file, 'w') as f:
            f.write(header)
            np.savetxt(f, data, fmt='%f %f %f %f')
