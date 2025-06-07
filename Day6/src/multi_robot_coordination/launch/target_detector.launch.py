from launch import LaunchDescription
from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='multi_robot_coordination',
#             executable='target_detector',
#             name='target_detector',
#             output='screen',
#             parameters=[{'num_robots': 4}]
#         ),

#         Node(
#             package='multi_robot_coordination',
#             executable='target_detector_gui',
#             name='target_detector_gui',
#             output='screen',
#             parameters=[{'num_robots': 4}]
#         )
#     ])

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='multi_robot_coordination',
            executable='target_detector',
            name='target_detector',
            output='screen',
            parameters=[{'num_robots': 4}]
        ),

        Node(
            package='multi_robot_coordination',
            executable='target_detector_gui',
            name='target_detector_gui',
            output='screen',
            parameters=[{'num_robots': 4}]
        )
    ])
