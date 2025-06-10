#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random

ROBOT_NAMES = ['tb1', 'tb2', 'tb3', 'tb4', 'tb4']  # Add more as needed

class MultiRobotController(Node):
    def __init__(self):
        super().__init__('multi_robot_controller')

        self.cmd_publishers = {}
        for name in ROBOT_NAMES:
            topic = f'/{name}/cmd_vel'
            self.cmd_publishers[name] = self.create_publisher(Twist, topic, 10)
            self.get_logger().info(f'Publisher created for {topic}')

        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.send_commands)

    def send_commands(self):
        for name in ROBOT_NAMES:
            twist = Twist()
            twist.linear.x = random.uniform(0.0, 0.2)
            twist.angular.z = random.uniform(-0.5, 0.5)
            self.cmd_publishers[name].publish(twist)
            self.get_logger().info(f'Sent command to {name}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
