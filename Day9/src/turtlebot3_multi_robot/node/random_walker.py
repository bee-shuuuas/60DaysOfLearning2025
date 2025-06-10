#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random

class RandomWalker(Node):
    def __init__(self):  # Fix: changed from **init** to __init__
        super().__init__('random_walker')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(2.0, self.move_randomly)
        
    def move_randomly(self):
        msg = Twist()
        msg.linear.x = random.uniform(0.1, 0.2)
        msg.angular.z = random.uniform(-1.0, 1.0)
        self.publisher.publish(msg)
        self.get_logger().info('Moving: Linear %.2f, Angular %.2f' % (msg.linear.x, msg.angular.z))

def main(args=None):
    rclpy.init(args=args)
    node = RandomWalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':  # Add this line to make it executable
    main()