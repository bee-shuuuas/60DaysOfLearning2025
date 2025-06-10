#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import math

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.map_data = None
        self.current_goal = None
        self.exploring = False
        
    def map_callback(self, msg):
        self.map_data = msg
        if not self.exploring:
            self.explore_frontiers()
    
    def find_frontiers(self):
        # Simplified frontier detection algorithm
        # A real implementation would be more sophisticated
        if self.map_data is None:
            return []
        
        # Convert map to numpy array for easier processing
        width = self.map_data.info.width
        height = self.map_data.info.height
        map_array = np.array(self.map_data.data).reshape(height, width)
        
        frontiers = []
        # Find frontiers (cells that are free and adjacent to unknown)
        for y in range(1, height-1):
            for x in range(1, width-1):
                # If cell is free (0)
                if map_array[y, x] == 0:
                    # Check neighboring cells for unknown (-1)
                    neighbors = [
                        map_array[y-1, x], map_array[y+1, x],
                        map_array[y, x-1], map_array[y, x+1]
                    ]
                    if -1 in neighbors:
                        # This is a frontier cell
                        world_x = x * self.map_data.info.resolution + self.map_data.info.origin.position.x
                        world_y = y * self.map_data.info.resolution + self.map_data.info.origin.position.y
                        frontiers.append((world_x, world_y))
        
        # Cluster frontiers and find centroids
        # (simplified here, would need proper clustering algorithm)
        return frontiers
    
    def explore_frontiers(self):
        frontiers = self.find_frontiers()
        if not frontiers:
            self.get_logger().info('No frontiers to explore')
            return
        
        # Select best frontier (e.g., closest or largest)
        # For simplicity, just pick the first one
        goal_x, goal_y = frontiers[0]
        
        # Navigate to the frontier
        self.navigate_to(goal_x, goal_y)
    
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
        self.nav_client.send_goal_async(goal_msg).add_done_callback(
            self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.exploring = False
            return
        
        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def get_result_callback(self, future):
        self.exploring = False
        self.get_logger().info('Reached frontier, looking for new frontiers')
        self.explore_frontiers()

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()