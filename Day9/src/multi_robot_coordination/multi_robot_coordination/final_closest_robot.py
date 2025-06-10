## Manual Code given to it; python3 src/multi_robot_coordination/multi_robot_coordination/final_closest_robot.py 1.46 -3.78 0

# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
# from nav_msgs.msg import Path
# from nav2_msgs.action import ComputePathToPose, NavigateToPose
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# import math
# import sys
# import time

# class MultiRobotPathCalculator(Node):
#     def __init__(self, robot_namespaces):
#         super().__init__('multi_robot_path_calculator')
#         self.robot_namespaces = robot_namespaces
        
#         # Enable simulation time
#         self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
#         # Create QoS profile for AMCL pose subscription
#         qos_profile = QoSProfile(
#             depth=50,
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL
#         )
        
#         # Dictionaries to store poses, pose messages, action clients, and path results
#         self.poses = {ns: None for ns in robot_namespaces}
#         self.latest_pose_msgs = {ns: None for ns in robot_namespaces}
#         self.path_action_clients = {}  # For ComputePathToPose
#         self.nav_action_clients = {}  # For NavigateToPose
#         self.path_results = {ns: None for ns in robot_namespaces}
        
#         # Create subscribers and action clients for each robot
#         for ns in robot_namespaces:
#             pose_topic = f'/{ns}/amcl_pose'
#             self.get_logger().info(f'Subscribing to {pose_topic}')
#             self.path_action_clients[ns] = ActionClient(self, ComputePathToPose, f'/{ns}/compute_path_to_pose')
#             self.nav_action_clients[ns] = ActionClient(self, NavigateToPose, f'/{ns}/navigate_to_pose')
#             self.create_subscription(
#                 PoseWithCovarianceStamped,
#                 pose_topic,
#                 lambda msg, namespace=ns: self.pose_callback(msg, namespace),
#                 qos_profile
#             )
        
#         # Store common goal
#         self.goal = None
#         self.goal_theta = 0.0
#         self.goal_received = False
        
#         self.get_logger().info(f'Multi Robot Path Calculator initialized for robots: {", ".join(robot_namespaces)}')
#         self.get_logger().info('Configured for simulation timing')

#     def pose_callback(self, msg, namespace):
#         """Callback to store the current robot pose for a given namespace"""
#         self.poses[namespace] = msg.pose.pose
#         self.latest_pose_msgs[namespace] = msg
#         self.get_logger().info(
#             f'Received /{namespace}/amcl_pose: '
#             f'Position=({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}), '
#             f'Orientation=(x={msg.pose.pose.orientation.x:.3f}, y={msg.pose.pose.orientation.y:.3f}, '
#             f'z={msg.pose.pose.orientation.z:.3f}, w={msg.pose.pose.orientation.w:.3f}), '
#             f'Timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
#         )

#     def calculate_path_length(self, path):
#         """Calculate the total length of a path"""
#         if not path.poses or len(path.poses) < 2:
#             return float('inf')
#         total_length = 0.0
#         for i in range(1, len(path.poses)):
#             prev_pose = path.poses[i-1].pose.position
#             curr_pose = path.poses[i].pose.position
#             dx = curr_pose.x - prev_pose.x
#             dy = curr_pose.y - prev_pose.y
#             dz = curr_pose.z - prev_pose.z
#             distance = math.sqrt(dx*dx + dy*dy + dz*dz)
#             total_length += distance
#         return total_length

#     def create_goal_pose(self, x, y, theta=0.0):
#         """Create a goal pose from x, y coordinates and optional orientation"""
#         goal_pose = PoseStamped()
#         goal_pose.header.frame_id = 'map'
#         goal_pose.header.stamp = self.get_clock().now().to_msg()
#         goal_pose.pose.position.x = x
#         goal_pose.pose.position.y = y
#         goal_pose.pose.position.z = 0.0
#         goal_pose.pose.orientation.x = 0.0
#         goal_pose.pose.orientation.y = 0.0
#         goal_pose.pose.orientation.z = math.sin(theta / 2.0)
#         goal_pose.pose.orientation.w = math.cos(theta / 2.0)
#         return goal_pose

#     def compute_path_to_goal(self, namespace, goal_x, goal_y, goal_theta=0.0):
#         """Compute path to goal for a specific robot and return path length"""
#         if self.poses[namespace] is None or self.latest_pose_msgs[namespace] is None:
#             self.get_logger().error(f'Robot pose not available for {namespace}!')
#             return None
            
#         # Check if action server is available
#         self.get_logger().info(f'Waiting for {namespace}/compute_path_to_pose action server...')
#         if not self.path_action_clients[namespace].wait_for_server(timeout_sec=10.0):
#             self.get_logger().error(f'Compute path action server not available for {namespace}!')
#             return None
            
#         # Create start pose
#         start_pose = PoseStamped()
#         start_pose.header.frame_id = 'map'
#         current_time = self.get_clock().now().to_msg()
#         pose_age = (current_time.sec - self.latest_pose_msgs[namespace].header.stamp.sec) + \
#                    (current_time.nanosec - self.latest_pose_msgs[namespace].header.stamp.nanosec) / 1e9
#         if pose_age < 5.0:
#             start_pose.header.stamp = self.latest_pose_msgs[namespace].header.stamp
#         else:
#             self.get_logger().warn(f'Latest pose timestamp for {namespace} is too old, using current time')
#             start_pose.header.stamp = current_time
#         start_pose.pose = self.poses[namespace]
        
#         # Create goal pose
#         goal_pose = self.create_goal_pose(goal_x, goal_y, goal_theta)
        
#         # Create action goal
#         action_goal = ComputePathToPose.Goal()
#         action_goal.goal = goal_pose
#         action_goal.start = start_pose
#         action_goal.planner_id = 'GridBased'
        
#         self.get_logger().info(f'Computing path for {namespace} from ({start_pose.pose.position.x:.2f}, {start_pose.pose.position.y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')
        
#         try:
#             future = self.path_action_clients[namespace].send_goal_async(action_goal)
#             rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
#             if future.result() is None:
#                 self.get_logger().error(f'Failed to get goal handle for {namespace} - action server timeout')
#                 return None
#             goal_handle = future.result()
#             if not goal_handle.accepted:
#                 self.get_logger().error(f'Goal was rejected by planner for {namespace}')
#                 return None
#             result_future = goal_handle.get_result_async()
#             rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
#             result = result_future.result()
#             if result is None:
#                 self.get_logger().error(f'Failed to compute path for {namespace} - planning timeout')
#                 return None
#             if not result.result.path.poses:
#                 self.get_logger().error(f'Planner returned empty path for {namespace}')
#                 return None
#         except Exception as e:
#             self.get_logger().error(f'Exception during path planning for {namespace}: {e}')
#             return None
            
#         path_length = self.calculate_path_length(result.result.path)
#         self.get_logger().info(f'Path computed successfully for {namespace}! Length: {path_length:.2f} meters, Poses: {len(result.result.path.poses)}')
#         return {
#             'path': result.result.path,
#             'length': path_length,
#             'num_poses': len(result.result.path.poses)
#         }

#     def send_navigation_goal(self, namespace, goal_x, goal_y, goal_theta=0.0):
#         """Send NavigateToPose action goal to the specified robot"""
#         self.get_logger().info(f'Sending navigation goal to {namespace}/navigate_to_pose')
        
#         # Check if action server is available
#         if not self.nav_action_clients[namespace].wait_for_server(timeout_sec=10.0):
#             self.get_logger().error(f'NavigateToPose action server not available for {namespace}!')
#             return False
            
#         # Create goal pose
#         goal_pose = self.create_goal_pose(goal_x, goal_y, goal_theta)
        
#         # Create action goal
#         action_goal = NavigateToPose.Goal()
#         action_goal.pose = goal_pose
        
#         self.get_logger().info(f'Navigating {namespace} to ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°)')
        
#         try:
#             future = self.nav_action_clients[namespace].send_goal_async(action_goal)
#             rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
#             if future.result() is None:
#                 self.get_logger().error(f'Failed to get goal handle for {namespace} - action server timeout')
#                 return False
#             goal_handle = future.result()
#             if not goal_handle.accepted:
#                 self.get_logger().error(f'Navigation goal was rejected for {namespace}')
#                 return False
#             result_future = goal_handle.get_result_async()
#             rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)  # Longer timeout for navigation
#             result = result_future.result()
#             if result is None:
#                 self.get_logger().error(f'Failed to navigate for {namespace} - navigation timeout')
#                 return False
#             self.get_logger().info(f'Navigation completed successfully for {namespace}!')
#             return True
#         except Exception as e:
#             self.get_logger().error(f'Exception during navigation for {namespace}: {e}')
#             return False

#     def wait_for_poses(self, timeout_sec=30.0):
#         """Wait for poses from all robots with timeout"""
#         self.get_logger().info(f'Waiting for poses from all robots (timeout: {timeout_sec}s)...')
#         start_time = time.time()
#         while rclpy.ok():
#             rclpy.spin_once(self, timeout_sec=0.1)
#             all_poses_received = all(self.poses[ns] is not None for ns in self.robot_namespaces)
#             if all_poses_received:
#                 self.get_logger().info('✓ Received poses for all robots')
#                 return True
#             elapsed = time.time() - start_time
#             if elapsed > timeout_sec:
#                 self.get_logger().error(f'✗ Timeout after {timeout_sec}s waiting for poses')
#                 for ns in self.robot_namespaces:
#                     if self.poses[ns] is None:
#                         self.get_logger().error(f'No pose received for {ns}')
#                 return False
#             if int(elapsed) % 5 == 0 and elapsed > 4:
#                 self.get_logger().info(f'  Still waiting... ({elapsed:.1f}s)')
#         return False

#     def find_closest_robot(self, goal_x, goal_y, goal_theta=0.0):
#         """Compute paths for all robots and return the one with the shortest path"""
#         for ns in self.robot_namespaces:
#             result = self.compute_path_to_goal(ns, goal_x, goal_y, goal_theta)
#             self.path_results[ns] = result
        
#         # Find the robot with the shortest path
#         min_length = float('inf')
#         closest_robot = None
#         for ns in self.robot_namespaces:
#             if self.path_results[ns] is not None and self.path_results[ns]['length'] < min_length:
#                 min_length = self.path_results[ns]['length']
#                 closest_robot = ns
        
#         return closest_robot, min_length

# def main(args=None):
#     rclpy.init(args=args)
    
#     # Define robot namespaces here (change as needed)
#     robot_namespaces = ['tb1', 'tb2', 'tb3']  # Modify this list to include desired robot namespaces
    
#     # Parse command-line arguments for goal coordinates
#     if len(sys.argv) >= 3:
#         try:
#             goal_x = float(sys.argv[1])
#             goal_y = float(sys.argv[2])
#             goal_theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
#         except ValueError:
#             print('Invalid arguments! Usage: python3 final_closest_robot.py <goal_x> <goal_y> [goal_theta]')
#             print('Example: python3 final_closest_robot.py 2.0 3.0 1.57')
#             rclpy.shutdown()
#             return
#     else:
#         print('Usage: python3 final_closest_robot.py <goal_x> <goal_y> [goal_theta]')
#         print('Example: python3 final_closest_robot.py 2.0 3.0 1.57')
#         rclpy.shutdown()
#         return
    
#     node = MultiRobotPathCalculator(robot_namespaces)
    
#     try:
#         # Wait for poses from all robots
#         if not node.wait_for_poses(timeout_sec=30.0):
#             node.get_logger().error('Could not get poses for all robots! Try:')
#             node.get_logger().error('  1. Ensure simulation is unpaused (ros2 service call /gazebo/unpause_physics std_srvs/srv/Empty)')
#             for ns in robot_namespaces:
#                 node.get_logger().error(f'  2. Set initial pose for {ns} in RViz or via /{ns}/initialpose')
#                 node.get_logger().error(f'  3. Verify AMCL localization for {ns} (/{ns}/particle_cloud in RViz)')
#                 node.get_logger().error(f'  4. Check /clock, /{ns}/scan, /{ns}/odom topics')
#             node.destroy_node()
#             rclpy.shutdown()
#             return
        
#         # Set goal from command-line arguments
#         node.goal = (goal_x, goal_y)
#         node.goal_theta = goal_theta
#         node.goal_received = True
#         node.get_logger().info(f'Goal set to ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°)')
        
#         # Compute paths and find the closest robot
#         closest_robot, min_length = node.find_closest_robot(goal_x, goal_y, goal_theta)
        
#         # Log path calculation results
#         node.get_logger().info(f'\n=== PATH CALCULATION RESULTS ===')
#         for ns in robot_namespaces:
#             if node.path_results[ns] is not None:
#                 node.get_logger().info(
#                     f'{ns}: Path length to goal ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°): '
#                     f'{node.path_results[ns]["length"]:.2f} meters, Poses: {node.path_results[ns]["num_poses"]}'
#                 )
#             else:
#                 node.get_logger().info(f'{ns}: Failed to compute path')
        
#         # Send navigation goal to the closest robot
#         if closest_robot is not None:
#             node.get_logger().info(f'\nClosest robot: {closest_robot} with path length {min_length:.2f} meters')
#             node.send_navigation_goal(closest_robot, goal_x, goal_y, goal_theta)
#         else:
#             node.get_logger().error('No valid paths computed for any robot!')
            
#     except Exception as e:
#         node.get_logger().error(f'Error during execution: {e}')
    
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Automatic Goal pose subscriber and publisher %%%%%%%%%%%%%%%%%%%%%%%%%%%% 

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time

class MultiRobotPathCalculator(Node):
    def __init__(self, robot_namespaces):
        super().__init__('multi_robot_path_calculator')
        self.robot_namespaces = robot_namespaces
        
        # Enable simulation time
        self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        # Create QoS profile for AMCL pose subscription
        amcl_qos_profile = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Create QoS profile for target_world_pose subscription
        target_qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE  # Changed to VOLATILE to match likely publisher QoS
        )
        
        # Dictionaries to store poses, pose messages, action clients, and path results
        self.poses = {ns: None for ns in robot_namespaces}
        self.latest_pose_msgs = {ns: None for ns in robot_namespaces}
        self.path_action_clients = {}  # For ComputePathToPose
        self.nav_action_clients = {}  # For NavigateToPose
        self.path_results = {ns: None for ns in robot_namespaces}
        
        # Create subscribers and action clients for each robot
        for ns in robot_namespaces:
            pose_topic = f'/{ns}/amcl_pose'
            self.get_logger().info(f'Subscribing to {pose_topic}')
            self.path_action_clients[ns] = ActionClient(self, ComputePathToPose, f'/{ns}/compute_path_to_pose')
            self.nav_action_clients[ns] = ActionClient(self, NavigateToPose, f'/{ns}/navigate_to_pose')
            self.create_subscription(
                PoseWithCovarianceStamped,
                pose_topic,
                lambda msg, namespace=ns: self.pose_callback(msg, namespace),
                amcl_qos_profile
            )
        
        # Subscribe to /target_world_pose topic
        self.create_subscription(
            PoseStamped,
            '/target_world_pose',
            self.target_pose_callback,
            target_qos_profile
        )
        
        # Store common goal
        self.goal = None
        self.goal_theta = 0.0
        self.goal_received = False
        
        self.get_logger().info(f'Multi Robot Path Calculator initialized for robots: {", ".join(robot_namespaces)}')
        self.get_logger().info('Configured for simulation timing')

    def target_pose_callback(self, msg):
        """Callback to store the goal pose from /target_world_pose"""
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.goal_theta = 0.0  # Default orientation, as in original code
        self.goal_received = True
        self.get_logger().info(
            f'Received /target_world_pose: '
            f'Position=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}), '
            f'Orientation=(x={msg.pose.orientation.x:.3f}, y={msg.pose.orientation.y:.3f}, '
            f'z={msg.pose.orientation.z:.3f}, w={msg.pose.orientation.w:.3f}), '
            f'Timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
        )

    def pose_callback(self, msg, namespace):
        """Callback to store the current robot pose for a given namespace"""
        self.poses[namespace] = msg.pose.pose
        self.latest_pose_msgs[namespace] = msg
        self.get_logger().info(
            f'Received /{namespace}/amcl_pose: '
            f'Position=({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}), '
            f'Orientation=(x={msg.pose.pose.orientation.x:.3f}, y={msg.pose.pose.orientation.y:.3f}, '
            f'z={msg.pose.pose.orientation.z:.3f}, w={msg.pose.pose.orientation.w:.3f}), '
            f'Timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
        )

    def calculate_path_length(self, path):
        """Calculate the total length of a path"""
        if not path.poses or len(path.poses) < 2:
            return float('inf')
        total_length = 0.0
        for i in range(1, len(path.poses)):
            prev_pose = path.poses[i-1].pose.position
            curr_pose = path.poses[i].pose.position
            dx = curr_pose.x - prev_pose.x
            dy = curr_pose.y - prev_pose.y
            dz = curr_pose.z - prev_pose.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += distance
        return total_length

    def create_goal_pose(self, x, y, theta=0.0):
        """Create a goal pose from x, y coordinates and optional orientation"""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_pose.pose.orientation.w = math.cos(theta / 2.0)
        return goal_pose

    def compute_path_to_goal(self, namespace, goal_x, goal_y, goal_theta=0.0):
        """Compute path to goal for a specific robot and return path length"""
        if self.poses[namespace] is None or self.latest_pose_msgs[namespace] is None:
            self.get_logger().error(f'Robot pose not available for {namespace}!')
            return None
            
        # Check if action server is available
        self.get_logger().info(f'Waiting for {namespace}/compute_path_to_pose action server...')
        if not self.path_action_clients[namespace].wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f'Compute path action server not available for {namespace}!')
            return None
            
        # Create start pose
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'map'
        current_time = self.get_clock().now().to_msg()
        pose_age = (current_time.sec - self.latest_pose_msgs[namespace].header.stamp.sec) + \
                   (current_time.nanosec - self.latest_pose_msgs[namespace].header.stamp.nanosec) / 1e9
        if pose_age < 5.0:
            start_pose.header.stamp = self.latest_pose_msgs[namespace].header.stamp
        else:
            self.get_logger().warn(f'Latest pose timestamp for {namespace} is too old, using current time')
            start_pose.header.stamp = current_time
        start_pose.pose = self.poses[namespace]
        
        # Create goal pose
        goal_pose = self.create_goal_pose(goal_x, goal_y, goal_theta)
        
        # Create action goal
        action_goal = ComputePathToPose.Goal()
        action_goal.goal = goal_pose
        action_goal.start = start_pose
        action_goal.planner_id = 'GridBased'
        
        self.get_logger().info(f'Computing path for {namespace} from ({start_pose.pose.position.x:.2f}, {start_pose.pose.position.y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')
        
        try:
            future = self.path_action_clients[namespace].send_goal_async(action_goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() is None:
                self.get_logger().error(f'Failed to get goal handle for {namespace} - action server timeout')
                return None
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f'Goal was rejected by planner for {namespace}')
                return None
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            result = result_future.result()
            if result is None:
                self.get_logger().error(f'Failed to compute path for {namespace} - planning timeout')
                return None
            if not result.result.path.poses:
                self.get_logger().error(f'Planner returned empty path for {namespace}')
                return None
        except Exception as e:
            self.get_logger().error(f'Exception during path planning for {namespace}: {e}')
            return None
            
        path_length = self.calculate_path_length(result.result.path)
        self.get_logger().info(f'Path computed successfully for {namespace}! Length: {path_length:.2f} meters, Poses: {len(result.result.path.poses)}')
        return {
            'path': result.result.path,
            'length': path_length,
            'num_poses': len(result.result.path.poses)
        }

    def send_navigation_goal(self, namespace, goal_x, goal_y, goal_theta=0.0):
        """Send NavigateToPose action goal to the specified robot"""
        self.get_logger().info(f'Sending navigation goal to {namespace}/navigate_to_pose')
        
        # Check if action server is available
        if not self.nav_action_clients[namespace].wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f'NavigateToPose action server not available for {namespace}!')
            return False
            
        # Create goal pose
        goal_pose = self.create_goal_pose(goal_x, goal_y, goal_theta)
        
        # Create action goal
        action_goal = NavigateToPose.Goal()
        action_goal.pose = goal_pose
        
        self.get_logger().info(f'Navigating {namespace} to ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°)')
        
        try:
            future = self.nav_action_clients[namespace].send_goal_async(action_goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if future.result() is None:
                self.get_logger().error(f'Failed to get goal handle for {namespace} - action server timeout')
                return False
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f'Navigation goal was rejected for {namespace}')
                return False
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)  # Longer timeout for navigation
            result = result_future.result()
            if result is None:
                self.get_logger().error(f'Failed to navigate for {namespace} - navigation timeout')
                return False
            self.get_logger().info(f'Navigation completed successfully for {namespace}!')
            return True
        except Exception as e:
            self.get_logger().error(f'Exception during navigation for {namespace}: {e}')
            return False

    def wait_for_poses(self, timeout_sec=30.0):
        """Wait for poses from all robots with timeout"""
        self.get_logger().info(f'Waiting for poses from all robots (timeout: {timeout_sec}s)...')
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            all_poses_received = all(self.poses[ns] is not None for ns in self.robot_namespaces)
            if all_poses_received:
                self.get_logger().info('✓ Received poses for all robots')
                return True
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                self.get_logger().error(f'✗ Timeout after {timeout_sec}s waiting for poses')
                for ns in self.robot_namespaces:
                    if self.poses[ns] is None:
                        self.get_logger().error(f'No pose received for {ns}')
                return False
            if int(elapsed) % 5 == 0 and elapsed > 4:
                self.get_logger().info(f'  Still waiting... ({elapsed:.1f}s)')
        return False

    def wait_for_goal(self, timeout_sec=30.0):
        """Wait for goal pose from /target_world_pose with timeout"""
        self.get_logger().info(f'Waiting for /target_world_pose (timeout: {timeout_sec}s)...')
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.goal_received:
                self.get_logger().info('✓ Received goal pose from /target_world_pose')
                return True
            elapsed = time.time() - start_time
            if elapsed > timeout_sec:
                self.get_logger().error(f'✗ Timeout after {timeout_sec}s waiting for /target_world_pose')
                return False
            if int(elapsed) % 5 == 0 and elapsed > 4:
                self.get_logger().info(f'  Still waiting for goal... ({elapsed:.1f}s)')
        return False

    def find_closest_robot(self, goal_x, goal_y, goal_theta=0.0):
        """Compute paths for all robots and return the one with the shortest path"""
        for ns in self.robot_namespaces:
            result = self.compute_path_to_goal(ns, goal_x, goal_y, goal_theta)
            self.path_results[ns] = result
        
        # Find the robot with the shortest path
        min_length = float('inf')
        closest_robot = None
        for ns in self.robot_namespaces:
            if self.path_results[ns] is not None and self.path_results[ns]['length'] < min_length:
                min_length = self.path_results[ns]['length']
                closest_robot = ns
        
        return closest_robot, min_length

def main(args=None):
    rclpy.init(args=args)
    
    # Define robot namespaces here (change as needed)
    robot_namespaces = ['tb1', 'tb2', 'tb3']  # Modify this list to include desired robot namespaces
    
    node = MultiRobotPathCalculator(robot_namespaces)
    
    try:
        # Wait for poses from all robots
        if not node.wait_for_poses(timeout_sec=30.0):
            node.get_logger().error('Could not get poses for all robots! Try:')
            node.get_logger().error('  1. Ensure simulation is unpaused (ros2 service call /gazebo/unpause_physics std_srvs/srv/Empty)')
            for ns in robot_namespaces:
                node.get_logger().error(f'  2. Set initial pose for {ns} in RViz or via /{ns}/initialpose')
                node.get_logger().error(f'  3. Verify AMCL localization for {ns} (/{ns}/particle_cloud in RViz)')
                node.get_logger().error(f'  4. Check /clock, /{ns}/scan, /{ns}/odom topics')
            node.destroy_node()
            rclpy.shutdown()
            return
        
        # Wait for goal from /target_world_pose
        if not node.wait_for_goal(timeout_sec=30.0):
            node.get_logger().error('Could not receive goal from /target_world_pose!')
            node.get_logger().error('  Ensure /target_world_pose topic is being published')
            node.destroy_node()
            rclpy.shutdown()
            return
        
        # Extract goal coordinates
        goal_x, goal_y = node.goal
        goal_theta = node.goal_theta
        node.get_logger().info(f'Goal set to ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°)')
        
        # Compute paths and find the closest robot
        closest_robot, min_length = node.find_closest_robot(goal_x, goal_y, goal_theta)
        
        # Log path calculation results
        node.get_logger().info(f'\n=== PATH CALCULATION RESULTS ===')
        for ns in robot_namespaces:
            if node.path_results[ns] is not None:
                node.get_logger().info(
                    f'{ns}: Path length to goal ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°): '
                    f'{node.path_results[ns]["length"]:.2f} meters, Poses: {node.path_results[ns]["num_poses"]}'
                )
            else:
                node.get_logger().info(f'{ns}: Failed to compute path')
        
        # Send navigation goal to the closest robot
        if closest_robot is not None:
            node.get_logger().info(f'\nClosest robot: {closest_robot} with path length {min_length:.2f} meters')
            node.send_navigation_goal(closest_robot, goal_x, goal_y, goal_theta)
        else:
            node.get_logger().error('No valid paths computed for any robot!')
            
    except Exception as e:
        node.get_logger().error(f'Error during execution: {e}')
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



# ############################### new distance publisher (/global_distance_to_target) publishes the minimum path length found during the closest robot calculation.
# # Publishes individual distances for each robot on /<robot_namespace>/distance_to_target
# # Publishes the global minimum distance on /global_distance_to_target

# # Published Topics:

# # /tb1/distance_to_target (Float64)
# # /tb2/distance_to_target (Float64)
# # /tb3/distance_to_target (Float64)

# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
# from nav_msgs.msg import Path
# from std_msgs.msg import Float64
# from nav2_msgs.action import ComputePathToPose, NavigateToPose
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# import math
# import time

# class MultiRobotPathCalculator(Node):
#     def __init__(self, robot_namespaces):
#         super().__init__('multi_robot_path_calculator')
#         self.robot_namespaces = robot_namespaces
        
#         # Enable simulation time
#         self.set_parameters([rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
#         # Create QoS profile for AMCL pose subscription
#         amcl_qos_profile = QoSProfile(
#             depth=50,
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL
#         )
        
#         # Create QoS profile for target_world_pose subscription
#         target_qos_profile = QoSProfile(
#             depth=10,
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # Create QoS profile for distance publishers
#         distance_qos_profile = QoSProfile(
#             depth=10,
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             durability=DurabilityPolicy.VOLATILE
#         )
        
#         # Dictionaries to store poses, pose messages, action clients, and path results
#         self.poses = {ns: None for ns in robot_namespaces}
#         self.latest_pose_msgs = {ns: None for ns in robot_namespaces}
#         self.path_action_clients = {}  # For ComputePathToPose
#         self.nav_action_clients = {}  # For NavigateToPose
#         self.path_results = {ns: None for ns in robot_namespaces}
        
#         # Create distance publishers for each robot
#         self.distance_publishers = {}
        
#         # Create subscribers, action clients, and distance publishers for each robot
#         for ns in robot_namespaces:
#             pose_topic = f'/{ns}/amcl_pose'
#             distance_topic = f'/{ns}/distance_to_target'
            
#             self.get_logger().info(f'Subscribing to {pose_topic}')
#             self.get_logger().info(f'Creating distance publisher for {distance_topic}')
            
#             # Create action clients
#             self.path_action_clients[ns] = ActionClient(self, ComputePathToPose, f'/{ns}/compute_path_to_pose')
#             self.nav_action_clients[ns] = ActionClient(self, NavigateToPose, f'/{ns}/navigate_to_pose')
            
#             # Create pose subscriber
#             self.create_subscription(
#                 PoseWithCovarianceStamped,
#                 pose_topic,
#                 lambda msg, namespace=ns: self.pose_callback(msg, namespace),
#                 amcl_qos_profile
#             )
            
#             # Create distance publisher
#             self.distance_publishers[ns] = self.create_publisher(
#                 Float64,
#                 distance_topic,
#                 distance_qos_profile
#             )
        
#         # Subscribe to /target_world_pose topic
#         self.create_subscription(
#             PoseStamped,
#             '/target_world_pose',
#             self.target_pose_callback,
#             target_qos_profile
#         )
        
#         # Store common goal
#         self.goal = None
#         self.goal_theta = 0.0
#         self.goal_received = False
        
#         self.get_logger().info(f'Multi Robot Path Calculator initialized for robots: {", ".join(robot_namespaces)}')
#         self.get_logger().info('Configured for simulation timing')

#     def target_pose_callback(self, msg):
#         """Callback to store the goal pose from /target_world_pose"""
#         self.goal = (msg.pose.position.x, msg.pose.position.y)
#         self.goal_theta = 0.0  # Default orientation, as in original code
#         self.goal_received = True
#         self.get_logger().info(
#             f'Received /target_world_pose: '
#             f'Position=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}), '
#             f'Orientation=(x={msg.pose.orientation.x:.3f}, y={msg.pose.orientation.y:.3f}, '
#             f'z={msg.pose.orientation.z:.3f}, w={msg.pose.orientation.w:.3f}), '
#             f'Timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
#         )

#     def pose_callback(self, msg, namespace):
#         """Callback to store the current robot pose for a given namespace"""
#         self.poses[namespace] = msg.pose.pose
#         self.latest_pose_msgs[namespace] = msg
#         self.get_logger().info(
#             f'Received /{namespace}/amcl_pose: '
#             f'Position=({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}), '
#             f'Orientation=(x={msg.pose.pose.orientation.x:.3f}, y={msg.pose.pose.orientation.y:.3f}, '
#             f'z={msg.pose.pose.orientation.z:.3f}, w={msg.pose.pose.orientation.w:.3f}), '
#             f'Timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
#         )

#     def publish_distance(self, namespace, distance):
#         """Publish distance to target for a specific robot"""
#         distance_msg = Float64()
#         distance_msg.data = distance
#         self.distance_publishers[namespace].publish(distance_msg)
#         self.get_logger().info(f'Published /{namespace}/distance_to_target = {distance:.2f}')

#     def calculate_path_length(self, path):
#         """Calculate the total length of a path"""
#         if not path.poses or len(path.poses) < 2:
#             return float('inf')
#         total_length = 0.0
#         for i in range(1, len(path.poses)):
#             prev_pose = path.poses[i-1].pose.position
#             curr_pose = path.poses[i].pose.position
#             dx = curr_pose.x - prev_pose.x
#             dy = curr_pose.y - prev_pose.y
#             dz = curr_pose.z - prev_pose.z
#             distance = math.sqrt(dx*dx + dy*dy + dz*dz)
#             total_length += distance
#         return total_length

#     def create_goal_pose(self, x, y, theta=0.0):
#         """Create a goal pose from x, y coordinates and optional orientation"""
#         goal_pose = PoseStamped()
#         goal_pose.header.frame_id = 'map'
#         goal_pose.header.stamp = self.get_clock().now().to_msg()
#         goal_pose.pose.position.x = x
#         goal_pose.pose.position.y = y
#         goal_pose.pose.position.z = 0.0
#         goal_pose.pose.orientation.x = 0.0
#         goal_pose.pose.orientation.y = 0.0
#         goal_pose.pose.orientation.z = math.sin(theta / 2.0)
#         goal_pose.pose.orientation.w = math.cos(theta / 2.0)
#         return goal_pose

#     def compute_path_to_goal(self, namespace, goal_x, goal_y, goal_theta=0.0):
#         """Compute path to goal for a specific robot and return path length"""
#         if self.poses[namespace] is None or self.latest_pose_msgs[namespace] is None:
#             self.get_logger().error(f'Robot pose not available for {namespace}!')
#             # Publish infinite distance for failed computation
#             self.publish_distance(namespace, float('inf'))
#             return None
            
#         # Check if action server is available
#         self.get_logger().info(f'Waiting for {namespace}/compute_path_to_pose action server...')
#         if not self.path_action_clients[namespace].wait_for_server(timeout_sec=10.0):
#             self.get_logger().error(f'Compute path action server not available for {namespace}!')
#             # Publish infinite distance for failed computation
#             self.publish_distance(namespace, float('inf'))
#             return None
            
#         # Create start pose
#         start_pose = PoseStamped()
#         start_pose.header.frame_id = 'map'
#         current_time = self.get_clock().now().to_msg()
#         pose_age = (current_time.sec - self.latest_pose_msgs[namespace].header.stamp.sec) + \
#                    (current_time.nanosec - self.latest_pose_msgs[namespace].header.stamp.nanosec) / 1e9
#         if pose_age < 5.0:
#             start_pose.header.stamp = self.latest_pose_msgs[namespace].header.stamp
#         else:
#             self.get_logger().warn(f'Latest pose timestamp for {namespace} is too old, using current time')
#             start_pose.header.stamp = current_time
#         start_pose.pose = self.poses[namespace]
        
#         # Create goal pose
#         goal_pose = self.create_goal_pose(goal_x, goal_y, goal_theta)
        
#         # Create action goal
#         action_goal = ComputePathToPose.Goal()
#         action_goal.goal = goal_pose
#         action_goal.start = start_pose
#         action_goal.planner_id = 'GridBased'
        
#         self.get_logger().info(f'Computing path for {namespace} from ({start_pose.pose.position.x:.2f}, {start_pose.pose.position.y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')
        
#         try:
#             future = self.path_action_clients[namespace].send_goal_async(action_goal)
#             rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
#             if future.result() is None:
#                 self.get_logger().error(f'Failed to get goal handle for {namespace} - action server timeout')
#                 self.publish_distance(namespace, float('inf'))
#                 return None
#             goal_handle = future.result()
#             if not goal_handle.accepted:
#                 self.get_logger().error(f'Goal was rejected by planner for {namespace}')
#                 self.publish_distance(namespace, float('inf'))
#                 return None
#             result_future = goal_handle.get_result_async()
#             rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
#             result = result_future.result()
#             if result is None:
#                 self.get_logger().error(f'Failed to compute path for {namespace} - planning timeout')
#                 self.publish_distance(namespace, float('inf'))
#                 return None
#             if not result.result.path.poses:
#                 self.get_logger().error(f'Planner returned empty path for {namespace}')
#                 self.publish_distance(namespace, float('inf'))
#                 return None
#         except Exception as e:
#             self.get_logger().error(f'Exception during path planning for {namespace}: {e}')
#             self.publish_distance(namespace, float('inf'))
#             return None
            
#         path_length = self.calculate_path_length(result.result.path)
#         self.get_logger().info(f'Path computed successfully for {namespace}! Length: {path_length:.2f} meters, Poses: {len(result.result.path.poses)}')
        
#         # Publish the distance immediately after calculation
#         self.publish_distance(namespace, path_length)
        
#         return {
#             'path': result.result.path,
#             'length': path_length,
#             'num_poses': len(result.result.path.poses)
#         }

#     def send_navigation_goal(self, namespace, goal_x, goal_y, goal_theta=0.0):
#         """Send NavigateToPose action goal to the specified robot"""
#         self.get_logger().info(f'Sending navigation goal to {namespace}/navigate_to_pose')
        
#         # Check if action server is available
#         if not self.nav_action_clients[namespace].wait_for_server(timeout_sec=10.0):
#             self.get_logger().error(f'NavigateToPose action server not available for {namespace}!')
#             return False
            
#         # Create goal pose
#         goal_pose = self.create_goal_pose(goal_x, goal_y, goal_theta)
        
#         # Create action goal
#         action_goal = NavigateToPose.Goal()
#         action_goal.pose = goal_pose
        
#         self.get_logger().info(f'Navigating {namespace} to ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°)')
        
#         try:
#             future = self.nav_action_clients[namespace].send_goal_async(action_goal)
#             rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
#             if future.result() is None:
#                 self.get_logger().error(f'Failed to get goal handle for {namespace} - action server timeout')
#                 return False
#             goal_handle = future.result()
#             if not goal_handle.accepted:
#                 self.get_logger().error(f'Navigation goal was rejected for {namespace}')
#                 return False
#             result_future = goal_handle.get_result_async()
#             rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)  # Longer timeout for navigation
#             result = result_future.result()
#             if result is None:
#                 self.get_logger().error(f'Failed to navigate for {namespace} - navigation timeout')
#                 return False
#             self.get_logger().info(f'Navigation completed successfully for {namespace}!')
#             return True
#         except Exception as e:
#             self.get_logger().error(f'Exception during navigation for {namespace}: {e}')
#             return False

#     def wait_for_poses(self, timeout_sec=30.0):
#         """Wait for poses from all robots with timeout"""
#         self.get_logger().info(f'Waiting for poses from all robots (timeout: {timeout_sec}s)...')
#         start_time = time.time()
#         while rclpy.ok():
#             rclpy.spin_once(self, timeout_sec=0.1)
#             all_poses_received = all(self.poses[ns] is not None for ns in self.robot_namespaces)
#             if all_poses_received:
#                 self.get_logger().info('✓ Received poses for all robots')
#                 return True
#             elapsed = time.time() - start_time
#             if elapsed > timeout_sec:
#                 self.get_logger().error(f'✗ Timeout after {timeout_sec}s waiting for poses')
#                 for ns in self.robot_namespaces:
#                     if self.poses[ns] is None:
#                         self.get_logger().error(f'No pose received for {ns}')
#                 return False
#             if int(elapsed) % 5 == 0 and elapsed > 4:
#                 self.get_logger().info(f'  Still waiting... ({elapsed:.1f}s)')
#         return False

#     def wait_for_goal(self, timeout_sec=30.0):
#         """Wait for goal pose from /target_world_pose with timeout"""
#         self.get_logger().info(f'Waiting for /target_world_pose (timeout: {timeout_sec}s)...')
#         start_time = time.time()
#         while rclpy.ok():
#             rclpy.spin_once(self, timeout_sec=0.1)
#             if self.goal_received:
#                 self.get_logger().info('✓ Received goal pose from /target_world_pose')
#                 return True
#             elapsed = time.time() - start_time
#             if elapsed > timeout_sec:
#                 self.get_logger().error(f'✗ Timeout after {timeout_sec}s waiting for /target_world_pose')
#                 return False
#             if int(elapsed) % 5 == 0 and elapsed > 4:
#                 self.get_logger().info(f'  Still waiting for goal... ({elapsed:.1f}s)')
#         return False

#     def find_closest_robot(self, goal_x, goal_y, goal_theta=0.0):
#         """Compute paths for all robots and return the one with the shortest path"""
#         self.get_logger().info('\n=== COMPUTING PATHS FOR ALL ROBOTS ===')
        
#         for ns in self.robot_namespaces:
#             self.get_logger().info(f'Computing path for {ns}...')
#             result = self.compute_path_to_goal(ns, goal_x, goal_y, goal_theta)
#             self.path_results[ns] = result
        
#         # Find the robot with the shortest path
#         min_length = float('inf')
#         closest_robot = None
#         for ns in self.robot_namespaces:
#             if self.path_results[ns] is not None and self.path_results[ns]['length'] < min_length:
#                 min_length = self.path_results[ns]['length']
#                 closest_robot = ns
        
#         return closest_robot, min_length

# def main(args=None):
#     rclpy.init(args=args)
    
#     # Define robot namespaces here (change as needed)
#     robot_namespaces = ['tb1', 'tb2', 'tb3']  # Modify this list to include desired robot namespaces
    
#     node = MultiRobotPathCalculator(robot_namespaces)
    
#     try:
#         # Wait for poses from all robots
#         if not node.wait_for_poses(timeout_sec=30.0):
#             node.get_logger().error('Could not get poses for all robots! Try:')
#             node.get_logger().error('  1. Ensure simulation is unpaused (ros2 service call /gazebo/unpause_physics std_srvs/srv/Empty)')
#             for ns in robot_namespaces:
#                 node.get_logger().error(f'  2. Set initial pose for {ns} in RViz or via /{ns}/initialpose')
#                 node.get_logger().error(f'  3. Verify AMCL localization for {ns} (/{ns}/particle_cloud in RViz)')
#                 node.get_logger().error(f'  4. Check /clock, /{ns}/scan, /{ns}/odom topics')
#             node.destroy_node()
#             rclpy.shutdown()
#             return
        
#         # Wait for goal from /target_world_pose
#         if not node.wait_for_goal(timeout_sec=30.0):
#             node.get_logger().error('Could not receive goal from /target_world_pose!')
#             node.get_logger().error('  Ensure /target_world_pose topic is being published')
#             node.destroy_node()
#             rclpy.shutdown()
#             return
        
#         # Extract goal coordinates
#         goal_x, goal_y = node.goal
#         goal_theta = node.goal_theta
#         node.get_logger().info(f'Goal set to ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°)')
        
#         # Compute paths and find the closest robot
#         closest_robot, min_length = node.find_closest_robot(goal_x, goal_y, goal_theta)
        
#         # Log path calculation results
#         node.get_logger().info(f'\n=== PATH CALCULATION RESULTS ===')
#         for ns in robot_namespaces:
#             if node.path_results[ns] is not None:
#                 node.get_logger().info(
#                     f'{ns}: Path length to goal ({goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_theta):.1f}°): '
#                     f'{node.path_results[ns]["length"]:.2f} meters, Poses: {node.path_results[ns]["num_poses"]}'
#                 )
#             else:
#                 node.get_logger().info(f'{ns}: Failed to compute path')
        
#         # Log published distances summary
#         node.get_logger().info(f'\n=== PUBLISHED DISTANCES SUMMARY ===')
#         for ns in robot_namespaces:
#             if node.path_results[ns] is not None:
#                 distance = node.path_results[ns]["length"]
#                 node.get_logger().info(f'/{ns}/distance_to_target = {distance:.2f}')
#             else:
#                 node.get_logger().info(f'/{ns}/distance_to_target = inf (failed)')
        
#         # Send navigation goal to the closest robot
#         if closest_robot is not None:
#             node.get_logger().info(f'\nClosest robot: {closest_robot} with path length {min_length:.2f} meters')
#             node.send_navigation_goal(closest_robot, goal_x, goal_y, goal_theta)
#         else:
#             node.get_logger().error('No valid paths computed for any robot!')
            
#     except Exception as e:
#         node.get_logger().error(f'Error during execution: {e}')
    
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()