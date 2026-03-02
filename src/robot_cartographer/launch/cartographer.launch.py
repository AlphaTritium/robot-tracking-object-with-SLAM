import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directory
    pkg_share = get_package_share_directory('robot_cartographer')
    
    # Launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    resolution = LaunchConfiguration('resolution', default='1.0')
    publish_period_sec = LaunchConfiguration('publish_period_sec', default='1.0')
    
    # Configuration paths
    configuration_directory = LaunchConfiguration(
        'configuration_directory',
        default=os.path.join(pkg_share, 'config')
    )
    configuration_basename = LaunchConfiguration(
        'configuration_basename',
        default='cartographer_2d.lua'
    )

    # Cartographer node
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', configuration_directory,
            '-configuration_basename', configuration_basename
        ],
        # Topic remappings - adjust based on your robot's topics
        remappings=[
            ('scan', '/scan'),                    # Lidar scan topic
            ('odom', '/odom'),                     # Odometry topic if available
            ('imu', '/imu'),                        # IMU topic if available
        ]
    )

    # Occupancy grid node (for map visualization)
    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-resolution', resolution,
            '-publish_period_sec', publish_period_sec
        ]
    )

    return LaunchDescription([
        cartographer_node,
        occupancy_grid_node
    ])