# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from multi_robot_coordination.msg import CylinderDetection
# from geometry_msgs.msg import Twist
# import time
# import signal

# class ExplorationNode(Node):
#     def __init__(self):
#         super().__init__('exploration_node')
#         self.get_logger().info("✅ Exploration node started")

#         # Define robot configuration - 3 robots
#         self.namespaces = ['tb1', 'tb2', 'tb3']

#         # Publishers and subscribers for each robot
#         self.cmd_vel_publishers = {}
#         self.exploration_vel_subscribers = {}
        
#         for namespace in self.namespaces:
#             self.cmd_vel_publishers[namespace] = self.create_publisher(
#                 Twist,
#                 f'{namespace}/cmd_vel',
#                 10
#             )
#             self.exploration_vel_subscribers[namespace] = self.create_subscription(
#                 Twist,
#                 f'{namespace}/cmd_vel/dd',
#                 lambda msg, ns=namespace: self.velocity_callback(msg, ns),
#                 10
#             )
#             self.get_logger().info(f"👂 Subscribed to {namespace}/cmd_vel/dd")
        
#         # Detection subscription
#         self.detection_sub = self.create_subscription(
#             CylinderDetection,
#             '/global_cylinder_detections',
#             self.detection_callback,
#             10
#         )
#         self.get_logger().info("👊 Subscribed to /global_cylinder_detections")
        
#         self.target_detected = False
#         self.shutdown_active = False
        
#         self.get_logger().info("🚖 Exploration node ready for robot exploration")

#     def velocity_callback(self, msg, namespace):
#         if self.shutdown_active or self.target_detected:
#             return
#         self.cmd_vel_publishers[namespace].publish(msg)
#         self.get_logger().debug(f"📡 Forwarded exploration velocity to {namespace}/cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")

#     def detection_callback(self, msg):
#         self.target_detected = msg.detected
#         if self.target_detected:
#             self.get_logger().info("🔍 Target detected, st  opping exploration")
#             for namespace in self.namespaces:
#                 self.publish_zero_velocity(namespace)
#         else:
#             self.get_logger().info("🔍 No target detected, continuing exploration")

#     def publish_zero_velocity(self, namespace):
#         twist = Twist()
#         for _ in range(3):
#             if namespace in self.cmd_vel_publishers:
#                 self.cmd_vel_publishers[namespace].publish(twist)
#         self.get_logger().info(f"✅ Sent zero velocity to {namespace}")

#     def shutdown_all(self):
#         self.get_logger().info("🛑 Shutting down exploration node...")
#         self.shutdown_active = True
#         for namespace in self.namespaces:
#             self.publish_zero_velocity(namespace)
#         self.get_logger().info("✅ Exploration node shutdown complete")

# def main(args=None):
#     rclpy.init(args=args)
#     node = ExplorationNode()
    
#     def shutdown_handler(signum, frame):
#         node.shutdown_all()
#         rclpy.shutdown()
#         print("👋 Exploration node shutdown!")
#         exit(0)
    
#     signal.signal(signal.SIGINT, shutdown_handler)
#     signal.signal(signal.SIGTERM, shutdown_handler)
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         shutdown_handler(None, None)

# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from multi_robot_coordination.msg import CylinderDetection
# from geometry_msgs.msg import Twist
# from std_msgs.msg import Bool
# import time
# import signal
# from enum import Enum

# class ExplorationState(Enum):
#     EXPLORING = "exploring"
#     TARGET_DETECTED = "target_detected"
#     WAITING_FOR_RESUME = "waiting_for_resume"

# class SmartExplorationNode(Node):
#     def __init__(self):
#         super().__init__('smart_exploration_node')
#         self.get_logger().info("✅ Smart Exploration Node started")

#         # Define robot configuration - 3 robots
#         self.namespaces = ['tb1', 'tb2', 'tb3']
#         self.current_state = ExplorationState.EXPLORING
#         self.target_detected = False
#         self.shutdown_active = False
        
#         # Timing for target lost management
#         self.target_lost_timer = None
#         self.target_lost_timeout = 3.0  # 3 seconds before resuming exploration
#         self.last_detection_time = None
        
#         # Publishers and subscribers for each robot
#         self.cmd_vel_publishers = {}
#         self.exploration_vel_subscribers = {}
        
#         for namespace in self.namespaces:
#             # Publisher to actual robot cmd_vel
#             self.cmd_vel_publishers[namespace] = self.create_publisher(
#                 Twist,
#                 f'{namespace}/cmd_vel',
#                 10
#             )
            
#             # Subscriber to exploration commands (from turtlebot3_drive)
#             self.exploration_vel_subscribers[namespace] = self.create_subscription(
#                 Twist,
#                 f'{namespace}/cmd_vel/dd',
#                 lambda msg, ns=namespace: self.velocity_callback(msg, ns),
#                 10
#             )
#             self.get_logger().info(f"👂 Subscribed to {namespace}/cmd_vel/dd")
        
#         # Detection subscription
#         self.detection_sub = self.create_subscription(
#             CylinderDetection,
#             '/global_cylinder_detections',
#             self.detection_callback,
#             10
#         )
#         self.get_logger().info("👊 Subscribed to /global_cylinder_detections")
        
#         # State publisher (for other nodes to know exploration status)
#         self.exploration_state_pub = self.create_publisher(
#             Bool, 
#             '/exploration_active', 
#             10
#         )
        
#         # Timer for periodic state updates
#         self.state_timer = self.create_timer(0.5, self.publish_exploration_state)
        
#         self.get_logger().info("🚖 Smart Exploration Node ready")

#     def velocity_callback(self, msg, namespace):
#         """Handle velocity commands from exploration system"""
#         if self.shutdown_active:
#             return
        
#         # Only forward exploration commands when actively exploring
#         if self.current_state == ExplorationState.EXPLORING:
#             self.cmd_vel_publishers[namespace].publish(msg)
#             self.get_logger().debug(f"📡 [EXPLORING] Forwarded to {namespace}: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
#         else:
#             # CRITICAL: Block exploration commands and send zero velocity instead
#             zero_twist = Twist()
#             self.cmd_vel_publishers[namespace].publish(zero_twist)
#             self.get_logger().debug(f"🚫 [BLOCKED] Exploration cmd blocked for {namespace}, sent zero velocity")

#     def detection_callback(self, msg):
#         """Handle target detection state changes"""
#         previous_detected = self.target_detected
#         self.target_detected = msg.detected
#         self.last_detection_time = self.get_clock().now()
        
#         if self.target_detected and not previous_detected:
#             # Target just detected - stop exploration immediately
#             self.get_logger().info("🔍 Target DETECTED! Stopping exploration")
#             self.transition_to_state(ExplorationState.TARGET_DETECTED)
            
#         elif not self.target_detected and previous_detected:
#             # Target just lost - start countdown to resume
#             self.get_logger().info("❌ Target LOST! Will resume exploration if not re-detected")
#             self.transition_to_state(ExplorationState.WAITING_FOR_RESUME)
#             self.start_resume_timer()
        
#         elif self.target_detected and self.current_state == ExplorationState.WAITING_FOR_RESUME:
#             # Target re-detected during wait period
#             self.get_logger().info("🔍 Target RE-DETECTED during wait period")
#             self.cancel_resume_timer()
#             self.transition_to_state(ExplorationState.TARGET_DETECTED)

#     def start_resume_timer(self):
#         """Start timer to resume exploration after target is lost"""
#         if self.target_lost_timer:
#             self.target_lost_timer.cancel()
        
#         self.target_lost_timer = self.create_timer(
#             self.target_lost_timeout,
#             self.resume_exploration_callback
#         )
#         self.get_logger().info(f"⏰ Resume timer started ({self.target_lost_timeout}s)")

#     def cancel_resume_timer(self):
#         """Cancel the resume timer"""
#         if self.target_lost_timer:
#             self.target_lost_timer.cancel()
#             self.target_lost_timer = None
#             self.get_logger().info("⏰ Resume timer cancelled")

#     def resume_exploration_callback(self):
#         """Resume exploration after timeout if target still not detected"""
#         if not self.target_detected:
#             self.get_logger().info("🔄 Resuming exploration - target still not detected")
#             self.transition_to_state(ExplorationState.EXPLORING)
#         else:
#             self.get_logger().info("🔍 Target re-detected, staying in detection mode")
#             self.transition_to_state(ExplorationState.TARGET_DETECTED)
        
#         # Clean up timer
#         if self.target_lost_timer:
#             self.target_lost_timer.cancel()
#             self.target_lost_timer = None

#     def transition_to_state(self, new_state):
#         """Handle state transitions"""
#         if new_state == self.current_state:
#             return
        
#         old_state = self.current_state
#         self.current_state = new_state
        
#         self.get_logger().info(f"🔄 State: {old_state.value} → {new_state.value}")
        
#         if new_state == ExplorationState.TARGET_DETECTED:
#             # Stop all robots immediately when target detected
#             self.stop_all_robots()
            
#         elif new_state == ExplorationState.EXPLORING:
#             self.get_logger().info("🚀 Exploration RESUMED - robots can move")
            
#         elif new_state == ExplorationState.WAITING_FOR_RESUME:
#             # Keep robots stopped while waiting
#             self.stop_all_robots()

#     def stop_all_robots(self):
#         """Send zero velocity to all robots and keep sending"""
#         twist = Twist()  # Zero velocity
#         for namespace in self.namespaces:
#             # Send multiple times to ensure delivery
#             for _ in range(5):
#                 if namespace in self.cmd_vel_publishers:
#                     self.cmd_vel_publishers[namespace].publish(twist)
#         self.get_logger().info("🛑 All robots stopped")
        
#         # Create a persistent stop timer that keeps sending zero velocity
#         if not hasattr(self, 'stop_timer') or self.stop_timer is None:
#             self.stop_timer = self.create_timer(0.1, self.continuous_stop_callback)  # 10Hz stop commands

#     def continuous_stop_callback(self):
#         """Continuously send zero velocity when not exploring"""
#         if self.current_state != ExplorationState.EXPLORING:
#             twist = Twist()  # Zero velocity
#             for namespace in self.namespaces:
#                 self.cmd_vel_publishers[namespace].publish(twist)
#         else:
#             # Cancel the stop timer when back to exploring
#             if hasattr(self, 'stop_timer') and self.stop_timer is not None:
#                 self.stop_timer.cancel()
#                 self.stop_timer = None

#     def publish_exploration_state(self):
#         """Publish current exploration state for other nodes"""
#         state_msg = Bool()
#         state_msg.data = (self.current_state == ExplorationState.EXPLORING)
#         self.exploration_state_pub.publish(state_msg)

#     def shutdown_all(self):
#         """Shutdown sequence"""
#         self.get_logger().info("🛑 Shutting down Smart Exploration Node...")
#         self.shutdown_active = True
#         self.stop_all_robots()
        
#         # Cancel any active timers
#         self.cancel_resume_timer()
#         if hasattr(self, 'stop_timer') and self.stop_timer is not None:
#             self.stop_timer.cancel()
#             self.stop_timer = None
        
#         self.get_logger().info("✅ Smart Exploration Node shutdown complete")

# def main(args=None):
#     rclpy.init(args=args)
#     node = SmartExplorationNode()
    
#     def shutdown_handler(signum, frame):
#         node.shutdown_all()
#         rclpy.shutdown()
#         print("👋 Smart Exploration Node shutdown!")
#         exit(0)
    
#     signal.signal(signal.SIGINT, shutdown_handler)
#     signal.signal(signal.SIGTERM, shutdown_handler)
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         shutdown_handler(None, None)

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from multi_robot_coordination.msg import CylinderDetection
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time
import signal
from enum import Enum

class ExplorationState(Enum):
    EXPLORING = "exploring"
    TARGET_DETECTED = "target_detected"
    WAITING_FOR_RESUME = "waiting_for_resume"

class SmartExplorationNode(Node):
    def __init__(self):
        super().__init__('smart_exploration_node')
        self.get_logger().info("✅ Smart Exploration Node started")

        # Define robot configuration - 3 robots
        self.namespaces = ['tb1', 'tb2', 'tb3']
        self.current_state = ExplorationState.EXPLORING
        self.target_detected = False
        self.shutdown_active = False
        
        # Timing for target lost management
        self.target_lost_timer = None
        self.target_lost_timeout = 3.0  # 3 seconds before resuming exploration
        self.last_detection_time = None
        
        # Publishers and subscribers for each robot
        self.cmd_vel_publishers = {}
        self.exploration_vel_subscribers = {}
        
        for namespace in self.namespaces:
            # Publisher to actual robot cmd_vel
            self.cmd_vel_publishers[namespace] = self.create_publisher(
                Twist,
                f'{namespace}/cmd_vel',
                10
            )
            
            # Subscriber to exploration commands (from turtlebot3_drive)
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
        
        # State publisher (for other nodes to know exploration status)
        self.exploration_state_pub = self.create_publisher(
            Bool, 
            '/exploration_active', 
            10
        )
        
        # Timer for periodic state updates
        self.state_timer = self.create_timer(0.5, self.publish_exploration_state)
        
        self.get_logger().info("🚖 Smart Exploration Node ready")

    def velocity_callback(self, msg, namespace):
        """Handle velocity commands from turtlebot3_drive exploration system"""
        if self.shutdown_active:
            return
        
        # Only forward exploration commands when actively exploring
        if self.current_state == ExplorationState.EXPLORING:
            # Forward turtlebot3_drive commands to actual cmd_vel
            self.cmd_vel_publishers[namespace].publish(msg)
            self.get_logger().debug(f"📡 [EXPLORING] Forwarded {namespace}: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
        else:
            # BLOCK turtlebot3_drive commands when target detected or waiting
            # DO NOT publish anything - let Nav2 have full control of cmd_vel
            self.get_logger().debug(f"🚫 [BLOCKED] turtlebot3_drive cmd blocked for {namespace} (state: {self.current_state.value})")

    def detection_callback(self, msg):
        """Handle target detection state changes"""
        previous_detected = self.target_detected
        self.target_detected = msg.detected
        self.last_detection_time = self.get_clock().now()
        
        self.get_logger().info(f"🔍 Detection: detected={msg.detected}, previous={previous_detected}, state={self.current_state.value}")
        
        if self.target_detected and not previous_detected:
            # Target just detected - stop exploration immediately
            self.get_logger().info("🔍 Target DETECTED! Blocking turtlebot3_drive, Nav2 takes control")
            self.transition_to_state(ExplorationState.TARGET_DETECTED)
            
        elif not self.target_detected and previous_detected:
            # Target just lost - start countdown to resume
            self.get_logger().info("❌ Target LOST! Starting countdown to resume exploration")
            self.transition_to_state(ExplorationState.WAITING_FOR_RESUME)
            self.start_resume_timer()
        
        elif self.target_detected and self.current_state == ExplorationState.WAITING_FOR_RESUME:
            # Target re-detected during wait period
            self.get_logger().info("🔍 Target RE-DETECTED during wait period")
            self.cancel_resume_timer()
            self.transition_to_state(ExplorationState.TARGET_DETECTED)

    def start_resume_timer(self):
        """Start timer to resume exploration after target is lost"""
        self.cancel_resume_timer()  # Clean up any existing timer first
        
        self.get_logger().info(f"⏰ Starting resume timer ({self.target_lost_timeout}s)")
        self.target_lost_timer = self.create_timer(
            self.target_lost_timeout,
            self.resume_exploration_callback
        )

    def cancel_resume_timer(self):
        """Cancel the resume timer"""
        if self.target_lost_timer is not None:
            self.get_logger().info("⏰ Cancelling resume timer")
            self.target_lost_timer.cancel()
            self.destroy_timer(self.target_lost_timer)  # Properly destroy the timer
            self.target_lost_timer = None

    def resume_exploration_callback(self):
        """Resume exploration after timeout if target still not detected"""
        self.get_logger().info(f"⏰ Resume timer fired! target_detected={self.target_detected}")
        
        # This is a one-shot timer, so clean up the reference first
        self.target_lost_timer = None
        
        if not self.target_detected:
            self.get_logger().info("🔄 RESUMING EXPLORATION - target still not detected")
            self.transition_to_state(ExplorationState.EXPLORING)
        else:
            self.get_logger().info("🔍 Target re-detected during timer, staying in detection mode")
            self.transition_to_state(ExplorationState.TARGET_DETECTED)

    def transition_to_state(self, new_state):
        """Handle state transitions"""
        if new_state == self.current_state:
            self.get_logger().debug(f"🔄 State unchanged: {new_state.value}")
            return
        
        old_state = self.current_state
        self.current_state = new_state
        
        self.get_logger().info(f"🔄 State transition: {old_state.value} → {new_state.value}")
        
        if new_state == ExplorationState.TARGET_DETECTED:
            self.get_logger().info("🎯 TARGET MODE: Stopping robots immediately, then Nav2 takes control")
            self.send_immediate_stop()
            
        elif new_state == ExplorationState.EXPLORING:
            self.get_logger().info("🚀 EXPLORATION MODE: turtlebot3_drive commands forwarded")
            
        elif new_state == ExplorationState.WAITING_FOR_RESUME:
            self.get_logger().info("⏳ WAITING MODE: turtlebot3_drive blocked, Nav2 controls robots")

    def send_immediate_stop(self):
        """Send immediate stop commands to prevent overshoot when target detected"""
        self.get_logger().info("🛑 Sending immediate stop commands to prevent overshoot")
        
        zero_twist = Twist()  # Zero velocity
        
        # Send stop commands multiple times for reliability
        for i in range(5):
            for namespace in self.namespaces:
                if namespace in self.cmd_vel_publishers:
                    self.cmd_vel_publishers[namespace].publish(zero_twist)
            
            # Small delay between sends to ensure delivery
            time.sleep(0.01)  # 10ms delay
        
        self.get_logger().info("🛑 Immediate stop completed - Nav2 can now take control")

    def publish_exploration_state(self):
        """Publish current exploration state for other nodes"""
        state_msg = Bool()
        state_msg.data = (self.current_state == ExplorationState.EXPLORING)
        self.exploration_state_pub.publish(state_msg)

    def shutdown_all(self):
        """Shutdown sequence"""
        self.get_logger().info("🛑 Shutting down Smart Exploration Node...")
        self.shutdown_active = True
        
        # Cancel any active timers
        self.cancel_resume_timer()
        
        self.get_logger().info("✅ Smart Exploration Node shutdown complete")

def main(args=None):
    rclpy.init(args=args)
    node = SmartExplorationNode()
    
    def shutdown_handler(signum, frame):
        node.shutdown_all()
        rclpy.shutdown()
        print("👋 Smart Exploration Node shutdown!")
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        shutdown_handler(None, None)

if __name__ == '__main__':
    main()