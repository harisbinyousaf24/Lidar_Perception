from Scripts.Utils.preprocessor_utils import PreprocessorUtils
from playground_utils import PlaygroundUtils

if __name__ == '__main__':
    input_ply_file = 'path_to_ply_file'
    output_ply_file = 'path_to_output_file'

    inlier_cloud, outlier_cloud = PreprocessorUtils.statistical_outlier_removal(input_ply_file
                                                                                ,[10, 3])

    inlier_pcd = PlaygroundUtils.pyntcloud_to_open3d(inlier_cloud)
    outlier_pcd = PlaygroundUtils.pyntcloud_to_open3d(outlier_cloud)

    outlier_pcd = PlaygroundUtils.color_pcd(outlier_pcd, 'r')

    PlaygroundUtils.visualize_pcd(inlier_pcd)
    PlaygroundUtils.visualize_multiple_pcds(inlier_pcd, outlier_pcd)