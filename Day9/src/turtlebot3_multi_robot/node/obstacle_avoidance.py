#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import time
import random

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.get_logger().info(f"Starting exploration node in namespace: {self.get_namespace()}")
        scan_topic = self.resolve_topic_name('scan')
        self.get_logger().info(f"Subscribing to LIDAR topic: {scan_topic}")
        self.subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist = Twist()
        self.min_distance = 0.5  # Minimum distance to start avoiding (meters)
        self.critical_distance = 0.3  # Distance to trigger backward movement (meters)
        self.linear_speed = 0.2  # Forward speed (m/s)
        self.backward_speed = -0.15  # Backward speed (m/s)
        self.angular_speed = 0.5  # Rotation speed (rad/s)
        self.initialized = False
        self.goal_direction = 'forward'  # Initial random direction
        self.directions = ['forward', 'left', 'right']  # Possible directions
        self.last_obstacle_time = self.get_clock().now()
        # Timer to update random direction
        self.timer = self.create_timer(3.0, self.update_goal_direction)
        self.get_logger().info("Waiting for LIDAR data...")
        time.sleep(2)  # Delay to ensure Gazebo is ready

    def update_goal_direction(self):
        # Check if stuck (repeated obstacles)
        current_time = self.get_clock().now()
        if (current_time - self.last_obstacle_time).nanoseconds / 1e9 < 2.0:
            # Recent obstacle, increase randomness to escape
            self.goal_direction = random.choice(self.directions)
            self.get_logger().info(f"Stuck detected, random direction: {self.goal_direction}")
        else:
            # Bias toward forward but allow exploration
            self.goal_direction = random.choices(
                self.directions, weights=[0.7, 0.15, 0.15], k=1)[0]
            self.get_logger().info(f"New random direction: {self.goal_direction}")

    def scan_callback(self, msg):
        if not self.initialized:
            self.get_logger().info(f"Received first LIDAR data in {self.get_namespace()}")
            self.initialized = True

        # Process LIDAR data
        ranges = msg.ranges
        # Front: ±30° (0° to 30°, 330° to 360°)
        front_ranges = ranges[-30:] + ranges[:30]
        # Left: 30° to 90°
        left_ranges = ranges[30:90]
        # Right: 270° to 330°
        right_ranges = ranges[270:330]
        
        # Filter valid ranges
        front_valid = [r for r in front_ranges if not math.isnan(r) and not math.isinf(r)]
        left_valid = [r for r in left_ranges if not math.isnan(r) and not math.isinf(r)]
        right_valid = [r for r in right_ranges if not math.isnan(r) and not math.isinf(r)]
        
        if not front_valid:
            self.get_logger().warn("No valid front LIDAR data, rotating")
            self.twist.linear.x = 0.0
            self.twist.angular.z = self.angular_speed
            self.last_obstacle_time = self.get_clock().now()
        else:
            min_front = min(front_valid)
            min_left = min(left_valid) if left_valid else float('inf')
            min_right = min(right_valid) if right_valid else float('inf')
            
            if min_front < self.critical_distance:
                self.get_logger().info(f"Critical obstacle at {min_front:.2f}m, moving backward")
                self.twist.linear.x = self.backward_speed
                self.twist.angular.z = 0.0
                self.last_obstacle_time = self.get_clock().now()
            elif min_front < self.min_distance:
                self.get_logger().info(f"Obstacle detected at {min_front:.2f}m, rotating")
                # Rotate away from closest side obstacle
                if min_left < min_right:
                    self.twist.angular.z = -self.angular_speed  # Turn right
                else:
                    self.twist.angular.z = self.angular_speed  # Turn left
                self.twist.linear.x = 0.0
                self.last_obstacle_time = self.get_clock().now()
            else:
                # Path clear, follow goal direction
                self.get_logger().info(f"Path clear, following direction: {self.goal_direction}")
                if self.goal_direction == 'forward':
                    self.twist.linear.x = self.linear_speed
                    self.twist.angular.z = 0.0
                elif self.goal_direction == 'left':
                    self.twist.linear.x = self.linear_speed * 0.7
                    self.twist.angular.z = self.angular_speed
                elif self.goal_direction == 'right':
                    self.twist.linear.x = self.linear_speed * 0.7
                    self.twist.angular.z = -self.angular_speed
                # Bias direction based on open space
                if min_left > min_right and min_left > self.min_distance:
                    self.goal_direction = 'left'
                    self.get_logger().info(f"Biased to left due to open space")
                elif min_right > min_left and min_right > self.min_distance:
                    self.goal_direction = 'right'
                    self.get_logger().info(f"Biased to right due to open space")

        self.publisher.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted, shutting down")
    finally:
        node.twist.linear.x = 0.0
        node.twist.angular.z = 0.0
        node.publisher.publish(node.twist)  # Stop the robot
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()