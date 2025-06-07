#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from multi_robot_coordination.msg import CylinderDetection
from geometry_msgs.msg import Twist
import time
import signal

class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')
        self.get_logger().info("✅ Exploration node started")

        # Define robot configuration - 3 robots
        self.namespaces = ['tb1', 'tb2', 'tb3']

        # Publishers and subscribers for each robot
        self.cmd_vel_publishers = {}
        self.exploration_vel_subscribers = {}
        
        for namespace in self.namespaces:
            self.cmd_vel_publishers[namespace] = self.create_publisher(
                Twist,
                f'{namespace}/cmd_vel',
                10
            )
            self.exploration_vel_subscribers[namespace] = self.create_subscription(
                Twist,
                f'{namespace}/cmd_vel/dd',
                lambda msg, ns=namespace: self.velocity_callback(msg, ns),
                10
            )
            self.get_logger().info(f"👂 Subscribed to {namespace}/cmd_vel/dd")
        
        # Detection subscription
        self.detection_sub = self.create_subscription(
            CylinderDetection,
            '/global_cylinder_detections',
            self.detection_callback,
            10
        )
        self.get_logger().info("👊 Subscribed to /global_cylinder_detections")
        
        self.target_detected = False
        self.shutdown_active = False
        
        self.get_logger().info("🚖 Exploration node ready for robot exploration")

    def velocity_callback(self, msg, namespace):
        if self.shutdown_active or self.target_detected:
            return
        self.cmd_vel_publishers[namespace].publish(msg)
        self.get_logger().debug(f"📡 Forwarded exploration velocity to {namespace}/cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")

    def detection_callback(self, msg):
        self.target_detected = msg.detected
        if self.target_detected:
            self.get_logger().info("🔍 Target detected, st  opping exploration")
            for namespace in self.namespaces:
                self.publish_zero_velocity(namespace)
        else:
            self.get_logger().info("🔍 No target detected, continuing exploration")

    def publish_zero_velocity(self, namespace):
        twist = Twist()
        for _ in range(3):
            if namespace in self.cmd_vel_publishers:
                self.cmd_vel_publishers[namespace].publish(twist)
        self.get_logger().info(f"✅ Sent zero velocity to {namespace}")

    def shutdown_all(self):
        self.get_logger().info("🛑 Shutting down exploration node...")
        self.shutdown_active = True
        for namespace in self.namespaces:
            self.publish_zero_velocity(namespace)
        self.get_logger().info("✅ Exploration node shutdown complete")

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    
    def shutdown_handler(signum, frame):
        node.shutdown_all()
        rclpy.shutdown()
        print("👋 Exploration node shutdown!")
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        shutdown_handler(None, None)

if __name__ == '__main__':
    main()