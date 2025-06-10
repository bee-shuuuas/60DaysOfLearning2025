import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from multi_robot_coordination.msg import CylinderDetection
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
from cv_bridge import CvBridge

class DepthToWorldConverter(Node):
    def __init__(self):
        super().__init__('depth_to_world_converter')

        self.current_namespace = None
        self.detection_sub = self.create_subscription(
            CylinderDetection,
            '/global_cylinder_detections',
            self.detection_callback,
            10
        )
        self.depth_sub = None
        self.camera_info_sub = None
        self.odom_sub = None
        self.get_logger().info("👊 Subscribed to /global_cylinder_detections")

        self.target_pub = self.create_publisher(PoseStamped, '/target_world_pose', 10)

        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.robot_pose = None
        self.focal_length_x = None
        self.focal_length_y = None
        self.image_center_x = None
        self.image_center_y = None

    def update_subscriptions(self, namespace):
        if self.depth_sub:
            self.destroy_subscription(self.depth_sub)
        if self.camera_info_sub:
            self.destroy_subscription(self.camera_info_sub)
        if self.odom_sub:
            self.destroy_subscription(self.odom_sub)

        self.current_namespace = namespace
        self.get_logger().info(f"Updating subscriptions for namespace: {namespace}")

        base_topic = f"{namespace}/intel_realsense_r200_depth"
        self.depth_sub = self.create_subscription(
            Image,
            f"{base_topic}/depth/image_raw",
            self.depth_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            f"{base_topic}/depth/camera_info",
            self.camera_info_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            f"{namespace}/odom",
            self.odom_callback,
            10
        )

    def camera_info_callback(self, msg):
        self.camera_info = msg
        if self.camera_info is not None:
            self.focal_length_x = self.camera_info.k[0]
            self.focal_length_y = self.camera_info.k[4]
            self.image_center_x = self.camera_info.k[2]
            self.image_center_y = self.camera_info.k[5]
            self.get_logger().info(f"Camera intrinsics: fx={self.focal_length_x}, fy={self.focal_length_y}, cx={self.image_center_x}, cy={self.image_center_y}")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def detection_callback(self, msg):
        self.get_logger().info(f"Received detection: namespace={msg.robot_namespace}, u={msg.u}, v={msg.v}, z={msg.z}, confidence={msg.confidence}")
        if not msg.detected:
            return

        if self.current_namespace != msg.robot_namespace:
            self.update_subscriptions(msg.robot_namespace)

        if self.depth_image is None or self.camera_info is None or self.robot_pose is None or \
           self.focal_length_x is None or self.focal_length_y is None or \
           self.image_center_x is None or self.image_center_y is None:
            self.get_logger().warn("Missing data")
            return

        u = msg.u
        v = msg.v
        confidence = msg.confidence

        if confidence < 0.5:
            self.get_logger().warn("Low confidence")
            return

        # Use depth from detection message
        depth = msg.z
        self.get_logger().info(f"Using depth from detection: {depth}")

        # CORRECTED: Standard camera frame coordinates
        # Camera frame: X-right, Y-down, Z-forward
        x_camera = (u - self.image_center_x) * depth / self.focal_length_x
        y_camera = (v - self.image_center_y) * depth / self.focal_length_y
        z_camera = depth
        
        self.get_logger().info(f"Camera frame: x={x_camera:.2f}, y={y_camera:.2f}, z={z_camera:.2f}")

        # CORRECTED: Transform from camera frame to robot frame
        # Assuming camera is mounted forward-facing on robot
        # Camera X (right) -> Robot Y (left, so negative)
        # Camera Y (down) -> Robot Z (up, so negative) 
        # Camera Z (forward) -> Robot X (forward)
        x_robot = z_camera  # Camera forward = Robot forward
        y_robot = -x_camera  # Camera right = Robot left (negative)
        z_robot = -y_camera  # Camera down = Robot up (negative)
        
        self.get_logger().info(f"Robot frame: x={x_robot:.2f}, y={y_robot:.2f}, z={z_robot:.2f}")

        # Get robot position and orientation
        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y
        quaternion = [
            self.robot_pose.orientation.x,
            self.robot_pose.orientation.y,
            self.robot_pose.orientation.z,
            self.robot_pose.orientation.w
        ]
        _, _, yaw = self.quaternion_to_euler(quaternion)
        self.get_logger().info(f"Robot pose: x={robot_x:.2f}, y={robot_y:.2f}, yaw={yaw:.2f}")

        # CORRECTED: Transform from robot frame to world frame
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Standard 2D rotation matrix
        x_world_relative = x_robot * cos_yaw - y_robot * sin_yaw
        y_world_relative = x_robot * sin_yaw + y_robot * cos_yaw
        
        self.get_logger().info(f"World relative: x={x_world_relative:.2f}, y={y_world_relative:.2f}")

        # Add robot's world position to get absolute world coordinates
        target_x = robot_x + x_world_relative
        target_y = robot_y + y_world_relative
        target_z = z_robot  # Assuming flat ground, could add robot_z if needed

        # Publish target pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'map'  # Match NAV2's expected frame
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = target_x
        target_pose.pose.position.y = target_y
        target_pose.pose.position.z = target_z
        target_pose.pose.orientation.w = 1.0

        self.target_pub.publish(target_pose)
        self.get_logger().info(f"Target world coordinates for {self.current_namespace}: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}")

    def quaternion_to_euler(self, quaternion):
        x, y, z, w = quaternion
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    node = DepthToWorldConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()