from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    num_robots = 4
    # num_robots = 4

    return LaunchDescription([
        Node(
            package='multi_robot_coordination',
            executable='central_coordinator',
            name='central_coordinator',
            output='screen',
            parameters=[{'num_robots': num_robots}]
        ),

        Node(
            package='multi_robot_coordination',  # same package if GUI is inside the same package
            executable='central_coordinator_gui',  # the GUI node executable name
            name='central_coordinator_gui',
            output='screen',
            parameters=[{'num_robots': num_robots}],
        ),
    ])
