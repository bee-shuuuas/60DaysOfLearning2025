# !/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import math
from tf2_ros import Buffer, TransformListener

class InfoGainExplorer(Node):
    def __init__(self):
        super().__init__('info_gain_explorer')
        
        # Create subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        
        # Setup navigation client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # TF setup for robot position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize variables
        self.map_data = None
        self.robot_position = (0.0, 0.0)
        self.exploring = False
        self.candidate_points = []
        
        # Parameters for exploration
        self.candidate_points_count = 20  # Number of random points to evaluate
        self.evaluation_radius = 3.0  # Radius to evaluate info gain around each point (meters)
        self.min_travel_distance = 1.0  # Minimum distance to travel for a new goal
    
    def map_callback(self, msg):
        self.map_data = msg
        if not self.exploring:
            self.find_best_exploration_point()
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_position = (x, y)
    
    def is_position_free(self, x, y):
        if self.map_data is None:
            return False
            
        # Convert world coordinates to map indices
        map_x = int((x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        map_y = int((y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        
        # Check if position is within map bounds and free
        if (0 <= map_x < self.map_data.info.width and 
            0 <= map_y < self.map_data.info.height):
            index = map_y * self.map_data.info.width + map_x
            return self.map_data.data[index] == 0  # 0 means free space
        return False
    
    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def generate_candidate_points(self):
        candidates = []
        
        if self.map_data is None:
            return candidates
            
        map_width = self.map_data.info.width * self.map_data.info.resolution
        map_height = self.map_data.info.height * self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        # Generate random points in the map
        attempts = 0
        while len(candidates) < self.candidate_points_count and attempts < 100:
            attempts += 1
            x = origin_x + np.random.uniform(0, map_width)
            y = origin_y + np.random.uniform(0, map_height)
            
            # Check if the point is in a free space and not too close to the robot
            if (self.is_position_free(x, y) and 
                self.distance((x, y), self.robot_position) > self.min_travel_distance):
                candidates.append((x, y))
                
        return candidates
    
    def calculate_info_gain(self, pos_x, pos_y):
        """
        Calculate information gain at a position by counting unknown cells
        within the evaluation radius
        """
        if self.map_data is None:
            return 0
            
        # Convert radius from meters to cells
        cell_radius = int(self.evaluation_radius / self.map_data.info.resolution)
        
        # Convert position to map indices
        center_x = int((pos_x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        center_y = int((pos_y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        
        # Count unknown cells within radius
        unknown_count = 0
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        for y in range(max(0, center_y - cell_radius), min(height, center_y + cell_radius + 1)):
            for x in range(max(0, center_x - cell_radius), min(width, center_x + cell_radius + 1)):
                # Check if cell is within the circular radius
                if ((x - center_x)**2 + (y - center_y)**2) <= cell_radius**2:
                    index = y * width + x
                    # -1 represents unknown space in most ROS maps
                    if self.map_data.data[index] == -1:
                        unknown_count += 1
        
        return unknown_count
    
    def find_best_exploration_point(self):
        candidates = self.generate_candidate_points()
        if not candidates:
            self.get_logger().info('No suitable exploration candidates found')
            return
            
        # Evaluate information gain for each candidate
        best_point = None
        best_gain = -1
        
        for point in candidates:
            gain = self.calculate_info_gain(*point)
            # Weigh gain by distance to prevent unnecessary long travels
            # (this is a simple heuristic, more sophisticated ones could be used)
            dist = self.distance(point, self.robot_position)
            weighted_gain = gain / (1 + 0.1 * dist)  # Simple weighting function
            
            if weighted_gain > best_gain:
                best_gain = weighted_gain
                best_point = point
        
        if best_point:
            self.navigate_to(*best_point)
        else:
            self.get_logger().info('Could not find a point with positive information gain')
    
    def navigate_to(self, x, y):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        self.exploring = True
        self.get_logger().info(f'Navigating to high info-gain point ({x:.2f}, {y:.2f})')
        
        self.nav_client.send_goal_async(goal_msg).add_done_callback(
            self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.exploring = False
            # Try to find another exploration point
            self.find_best_exploration_point()
            return
        
        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        self.exploring = False
        self.get_logger().info('Reached exploration point, finding next best point')
        self.find_best_exploration_point()

def main(args=None):
    rclpy.init(args=args)
    node = InfoGainExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()