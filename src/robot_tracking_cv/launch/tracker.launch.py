import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('robot_tracking_cv'),
        'config',
        'tracking_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='robot_tracking_cv',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[config]
        )
    ])