import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import csv
from datetime import datetime

class MapSubscriber(Node):

    def __init__(self):
        super().__init__('map_subscriber')
        # Create subscription to the /map topic
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Log info about the start of the subscriber
        self.get_logger().info("Map subscriber node has started.")

    def map_callback(self, msg):
        # Log incoming map data information
        self.get_logger().info("Received new map data.")

        # Define the step size (in meters)
        step_size = 0.5  # step size in meters, for this case 0.5 meter grid
        resolution = msg.info.resolution  # map resolution (e.g., 0.05 meters per cell)
        width = msg.info.width
        height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Log map info such as width, height, resolution, etc.
        self.get_logger().info(f"Map Resolution: {resolution}m, Width: {width}, Height: {height}")
        self.get_logger().info(f"Map Origin: ({origin_x}, {origin_y})")

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Open CSV file in append mode
        try:
            with open('map_coordinates.csv', 'a', newline='') as file:
                writer = csv.writer(file)

                # If the file is empty, write the header
                if file.tell() == 0:
                    writer.writerow(['Timestamp', 'X', 'Y', 'Occupancy Value', 'Status'])
                    self.get_logger().info("Header written to the CSV file.")

                # Loop through all cells in the grid
                for i in range(height):
                    for j in range(width):
                        # Convert grid indices (i, j) to real-world coordinates (x, y)
                        world_x = origin_x + j * resolution
                        world_y = origin_y + i * resolution

                        # Debugging log: check if the condition is being met
                        if (world_x % step_size == 0) and (world_y % step_size == 0):
                            # Get the occupancy value for this grid cell
                            cell_value = msg.data[i * width + j]

                            # Classify the cell as Free, Occupied, or Unknown
                            if cell_value == 0:
                                status = "Free"
                            elif cell_value == 100:
                                status = "Occupied"
                            else:
                                status = "Unknown"

                            # Log the point being processed
                            self.get_logger().info(f"Writing point at ({world_x}, {world_y}) - {status}")

                            # Write the data including timestamp
                            writer.writerow([timestamp, world_x, world_y, cell_value, status])

                self.get_logger().info("Map data saved to map_coordinates.csv")
        
        except Exception as e:
            self.get_logger().error(f"Error while writing to CSV: {e}")

def main(args=None):
    rclpy.init(args=args)

    map_subscriber = MapSubscriber()

    # Log that the node is running
    map_subscriber.get_logger().info("Node is spinning... waiting for map data.")

    # Spin to keep the node running
    rclpy.spin(map_subscriber)

    # Shutdown the ROS 2 Python client
    map_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
