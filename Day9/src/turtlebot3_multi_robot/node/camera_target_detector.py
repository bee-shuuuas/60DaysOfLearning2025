import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraTargetDetector(Node):
    def __init__(self):
        super().__init__('camera_target_detector')

        # Declare & get the robot name
        self.declare_parameter('robot_name', 'tb1')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        # Topic based on robot name
        image_topic = f'/{self.robot_name}/camera/image_raw'

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info(f"Started target detection for robot: {self.robot_name} on {image_topic}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Dummy target detection (red color example)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,100,100), (10,255,255))
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Check for presence
        if cv2.countNonZero(mask) > 500:
            self.get_logger().info(f"{self.robot_name}: Target Detected!")

def main(args=None):
    rclpy.init(args=args)
    node = CameraTargetDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
