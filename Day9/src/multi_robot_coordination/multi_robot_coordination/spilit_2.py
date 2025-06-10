#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from multi_robot_coordination.msg import CylinderDetection
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time
import threading
import tf_transformations
import signal
import numpy as np

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.get_logger().info("✅ Control node started")

        # Define robot configuration - 3 robots
        self.namespaces = ['/tb1', '/tb2', '/tb3']

        # Publishers and subscribers for each robot
        self.cmd_vel_publishers = {}
        self.control_vel_subscribers = {}
        self.odom_subscribers = {}
        self.controllers = {}
        
        for namespace in self.namespaces:
            self.cmd_vel_publishers[namespace] = self.create_publisher(
                Twist,
                f'{namespace}/cmd_vel',
                10
            )
            self.control_vel_subscribers[namespace] = self.create_subscription(
                Twist,
                f'{namespace}/cmd_vel/control',
                lambda msg, ns=namespace: self.velocity_callback(msg, ns),
                10
            )
            self.odom_subscribers[namespace] = self.create_subscription(
                Odometry,
                f'{namespace}/odom',
                lambda msg, ns=namespace: self.odom_callback(msg, ns),
                10
            )
            self.get_logger().info(f"👂 Subscribed to {namespace}/cmd_vel/control and {namespace}/odom")
        
        # Detection subscription
        self.detection_sub = self.create_subscription(
            CylinderDetection,
            '/global_cylinder_detections',
            self.detection_callback,
            10
        )
        self.get_logger().info("👊 Subscribed to /global_cylinder_detections")
        
        self.target_detected = False
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.x_min = 0.0
        self.y_min = 0.0
        self.x_max = 0.0
        self.y_max = 0.0
        self.image_width = 640.0
        self.assigned_robot = None
        self.shutdown_active = False
        self.pause_start_time = None
        self.pause_duration = 5.0
        self.target_reached = False
        
        # Robot positions (optional, not used)
        self.robot_positions = {
            '/tb0_0': (-2.0, -2.0),
            '/tb0_1': (0.0, -2.0),
            '/tb1_0': (-2.0, 0.0),
            '/tb1_1': (0.0, 0.0)
        }
        
        # Robot odometry data
        self.robot_states = {
            namespace: {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'odom_received': False}
            for namespace in self.namespaces
        }
        
        self.get_logger().info("🚖 Control node ready for target tracking")
        
        # Start monitoring threads
        threading.Thread(target=self.monitor_pause_state, daemon=True).start()
        threading.Thread(target=self.check_control_timeout, daemon=True).start()

    def velocity_callback(self, msg, namespace):
        if self.shutdown_active or self.pause_start_time is not None or self.target_reached:
            return
        if namespace == self.assigned_robot and self.target_detected:
            self.cmd_vel_publishers[namespace].publish(msg)
            self.get_logger().debug(f"📡 Forwarded control velocity to {namespace}/cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")

    def odom_callback(self, msg, namespace):
        self.robot_states[namespace]['x'] = msg.pose.pose.position.x
        self.robot_states[namespace]['y'] = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf_transformations.euler_from_quaternion(quaternion)
        self.robot_states[namespace]['yaw'] = euler[2]
        self.robot_states[namespace]['odom_received'] = True

    def monitor_pause_state(self):
        while not self.shutdown_active:
            if self.pause_start_time is not None:
                elapsed = time.time() - self.pause_start_time
                if elapsed >= self.pause_duration:
                    self.get_logger().info(f"⏰ Pause period ended after {elapsed:.1f} seconds")
                    self.pause_start_time = None
                    self.assign_robot_to_target()
            time.sleep(0.1)

    def check_control_timeout(self):
        while not self.shutdown_active:
            for namespace in self.namespaces:
                if namespace in self.controllers and self.controllers[namespace]['start_time'] is not None:
                    if time.time() - self.controllers[namespace]['start_time'] > 60.0:
                        self.get_logger().warn(f"⏰ Control timeout for {namespace}, stopping control")
                        self.stop_control(namespace)
            time.sleep(1.0)

    def publish_zero_velocity_continuous(self, namespace, duration=1.0):
        twist = Twist()
        start_time = time.time()
        count = 0
        while time.time() - start_time < duration and not self.shutdown_active:
            if namespace in self.cmd_vel_publishers:
                self.cmd_vel_publishers[namespace].publish(twist)
                count += 1
            time.sleep(0.05)
        self.get_logger().info(f"✅ Published {count} zero velocity commands to {namespace}")

    def start_control(self, namespace, target_x, target_y, target_z, x_min, y_min, x_max, y_max, image_width):
        self.get_logger().info(f"🎯 Starting control for {namespace}")
        try:
            if namespace in self.controllers:
                self.stop_control(namespace)
            
            self.publish_zero_velocity_continuous(namespace, duration=0.2)
            
            self.controllers[namespace] = {
                'controller': GrokControlImpl(self, namespace, target_x, target_y, target_z, x_min, y_min, x_max, y_max, image_width),
                'start_time': time.time()
            }
            self.get_logger().info(f"✅ Control started for {namespace}")
            self.target_reached = False
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to start control for {namespace}: {e}")
            self.stop_control(namespace)

    def stop_control(self, namespace):
        self.get_logger().info(f"🚖 Stopping control for {namespace}")
        if namespace in self.controllers:
            self.controllers[namespace]['controller'].stop()
            del self.controllers[namespace]
        self.publish_zero_velocity_continuous(namespace, duration=0.5)

    def assign_robot_to_target(self):
        if not self.target_detected or not self.assigned_robot or self.target_reached:
            return
        self.get_logger().info(f"🎯 Target at ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), bbox=({self.x_min:.0f}, {self.y_min:.0f}, {self.x_max:.0f}, {self.y_max:.0f})")
        self.start_control(self.assigned_robot, self.target_x, self.target_y, self.target_z, self.x_min, self.y_min, self.x_max, self.y_max, self.image_width)

    def detection_callback(self, msg):
        if not msg.detected:
            if self.target_detected:
                self.get_logger().info("🔍 No target detected, stopping control")
                self.target_detected = False
                self.assigned_robot = None
                self.pause_start_time = None
                self.target_reached = False
                for namespace in self.namespaces:
                    self.stop_control(namespace)
            return

        robot_namespace = (msg.robot_namespace or msg.header.frame_id).strip()
        if not robot_namespace.startswith('/'):
            robot_namespace = f'/{robot_namespace}'
        if robot_namespace not in self.namespaces:
            self.get_logger().warn(f"⚠️ Unknown robot namespace: '{robot_namespace}'")
            return

        self.get_logger().info(f"🎯 DETECTION from {robot_namespace}: confidence={msg.confidence:.2f}, depth={msg.z:.3f}m")

        if self.target_detected and robot_namespace == self.assigned_robot and self.pause_start_time is None and not self.target_reached:
            self.target_x = msg.x
            self.target_y = msg.y
            self.target_z = msg.z
            self.x_min = msg.x_min
            self.y_min = msg.y_min
            self.x_max = msg.x_max
            self.y_max = msg.y_max
            self.image_width = msg.image_width if msg.image_width > 0 else 640.0
            self.get_logger().info(f"📍 Updated target for {self.assigned_robot} to ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), bbox=({self.x_min:.0f}, {self.y_min:.0f}, {self.x_max:.0f}, {self.y_max:.0f})")
            if self.assigned_robot in self.controllers:
                self.controllers[self.assigned_robot]['controller'].update_target(self.target_x, self.target_y, self.target_z, self.x_min, self.y_min, self.x_max, self.y_max, self.image_width)
            return

        if not self.target_detected:
            self.get_logger().info(f"⏸️ Starting {self.pause_duration} second pause")
            self.target_detected = True
            self.target_x = msg.x
            self.target_y = msg.y
            self.target_z = msg.z
            self.x_min = msg.x_min
            self.y_min = msg.y_min
            self.x_max = msg.x_max
            self.y_max = msg.y_max
            self.image_width = msg.image_width if msg.image_width > 0 else 640.0
            self.assigned_robot = robot_namespace
            self.pause_start_time = time.time()

    def shutdown_all(self):
        self.get_logger().info("🛑 Shutting down control node...")
        self.shutdown_active = True
        for namespace in self.namespaces:
            self.stop_control(namespace)
        for _ in range(10):
            for namespace in self.namespaces:
                twist = Twist()
                if namespace in self.cmd_vel_publishers:
                    self.cmd_vel_publishers[namespace].publish(twist)
            time.sleep(0.05)
        self.get_logger().info("✅ Control node shutdown complete")

class GrokControlImpl:
    def __init__(self, parent_node, namespace, target_x, target_y, target_z, x_min, y_min, x_max, y_max, image_width):
        self.parent = parent_node
        self.namespace = namespace
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.target_z = float(target_z)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.image_width = float(image_width)
        self.cmd_vel_pub = parent_node.cmd_vel_publishers[namespace]
        
        # Control parameters
        self.depth_threshold = 0.3
        self.max_linear_velocity = 0.25
        self.min_linear_velocity = 0.05
        self.max_angular_velocity = 0.4
        
        # Improved centering parameters
        self.center_threshold_tight = 10.0  # pixels - very tight centering
        self.center_threshold_loose = 30.0  # pixels - loose centering for approach
        self.approach_velocity = 0.08
        self.centering_velocity = 0.05  # slow forward motion while centering

        # Enhanced PD Controller Gains for better centering
        self.Kp_linear = 0.8
        self.Kd_linear = 0.15
        self.Kp_angular = 0.003  # Increased for better centering response
        self.Kd_angular = 0.001  # Increased for smoother centering
        
        # Centering-specific gains
        self.Kp_center = 0.004  # Stronger centering response
        self.Kd_center = 0.0015
        self.max_center_angular = 0.6  # Allow faster centering rotation

        # State
        self.target_reached = False
        self.is_centered = False
        self.centering_stable_count = 0
        self.centering_stable_threshold = 5  # Number of consecutive centered readings
        self.last_error_linear = 0.0
        self.last_error_angle = 0.0
        self.last_time = time.time()

        # Timer for control loop
        self.timer = parent_node.create_timer(0.1, self.control_loop)
        x_center = (self.x_min + self.x_max) / 2
        image_center = self.image_width / 2
        center_error = abs(x_center - image_center)
        self.parent.get_logger().info(f"🎯 Target for {namespace}: ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), center_error={center_error:.1f}px")

    def update_target(self, target_x, target_y, target_z, x_min, y_min, x_max, y_max, image_width):
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.target_z = float(target_z)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.image_width = float(image_width)
        self.target_reached = False
        # Reset centering state when target updates
        self.is_centered = False
        self.centering_stable_count = 0
        x_center = (self.x_min + self.x_max) / 2
        image_center = self.image_width / 2
        center_error = abs(x_center - image_center)
        self.parent.get_logger().info(f"🎯 Updated target for {self.namespace}: ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), center_error={center_error:.1f}px")

    def is_target_centered(self, error_x):
        """Check if target is properly centered"""
        return abs(error_x) <= self.center_threshold_tight

    def control_loop(self):
        if self.target_reached or not self.parent.robot_states[self.namespace]['odom_received']:
            return

        # Calculate target center and error
        x_center = (self.x_min + self.x_max) / 2
        image_center = self.image_width / 2
        error_x = x_center - image_center
        depth = self.target_z

        # Get current time
        current_time = time.time()
        dt = current_time - self.last_time if current_time > self.last_time else 0.1

        cmd = Twist()
        
        # Check if target is reached based on depth (only if centered)
        if (depth < self.depth_threshold and not np.isinf(depth) and not np.isnan(depth) 
            and self.is_target_centered(error_x)):
            self.parent.get_logger().info(f"🎉 Target reached for {self.namespace}! Depth: {depth:.3f}m, centered!")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            for _ in range(5):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.05)
            self.target_reached = True
            self.parent.target_reached = True
            return

        # Phase 1: Precise Centering (Priority)
        if not self.is_target_centered(error_x):
            # Use enhanced PD controller for centering
            derivative_error_angle = (error_x - self.last_error_angle) / dt
            angular_vel = -self.Kp_center * error_x - self.Kd_center * derivative_error_angle
            angular_vel = max(min(angular_vel, self.max_center_angular), -self.max_center_angular)
            
            # Very slow forward motion while centering (only if not too far off)
            if abs(error_x) < self.center_threshold_loose:
                cmd.linear.x = self.centering_velocity
            else:
                cmd.linear.x = 0.0
            
            cmd.angular.z = angular_vel
            
            self.is_centered = False
            self.centering_stable_count = 0
            
            self.parent.get_logger().info(f"🎯 CENTERING {self.namespace}: error_x={error_x:.1f}px, angular={angular_vel:.3f}, linear={cmd.linear.x:.3f}")
            
        # Phase 2: Centered - Now approach or maintain position
        else:
            # Target is centered - increment stable counter
            self.centering_stable_count += 1
            if self.centering_stable_count >= self.centering_stable_threshold:
                self.is_centered = True
            
            # Approach logic when centered
            if np.isinf(depth) or np.isnan(depth):
                # Unknown depth but centered - approach slowly
                cmd.linear.x = self.approach_velocity
                cmd.angular.z = 0.0
                self.parent.get_logger().info(f"➡️ CENTERED APPROACH {self.namespace}: unknown depth, moving at {self.approach_velocity:.2f}m/s")
            else:
                # Known depth and centered - use PD control for approach
                linear_error = depth
                derivative_error_linear = (linear_error - self.last_error_linear) / dt
                linear_vel = self.Kp_linear * linear_error + self.Kd_linear * derivative_error_linear
                linear_vel = max(min(linear_vel, self.max_linear_velocity), self.min_linear_velocity)
                
                cmd.linear.x = linear_vel
                cmd.angular.z = 0.0
                
                self.parent.get_logger().info(f"➡️ CENTERED APPROACH {self.namespace}: depth={depth:.2f}m, stable_count={self.centering_stable_count}, linear={linear_vel:.3f}")
                
                self.last_error_linear = linear_error

        # Update error tracking
        self.last_error_angle = error_x
        self.last_time = current_time

        # Publish command
        self.cmd_vel_pub.publish(cmd)

    def stop(self):
        cmd = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(cmd)
        self.parent.get_logger().info(f"🛑 Robot {self.namespace} stopped")
        if hasattr(self, 'timer'):
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    
    def shutdown_handler(signum, frame):
        node.shutdown_all()
        rclpy.shutdown()
        print("👋 Control node shutdown!")
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        shutdown_handler(None, None)

if __name__ == '__main__':
    main()