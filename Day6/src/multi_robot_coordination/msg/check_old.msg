# # CylinderDetection.msg
# bool detected
# float32 u  # Center x-coordinate in pixels
# float32 v  # Center y-coordinate in pixels
# float32 x  # 3D x-coordinate (meters)
# float32 y  # 3D y-coordinate (meters)
# float32 z  # 3D z-coordinate (meters)
# float32 confidence  # Detection confidence
# float32[] xyxy  # Bounding box coordinates [x1, y1, x2, y2]

std_msgs/Header header
# bool detected
# float64 u
# float64 v
# float64 x
# float64 y
# float64 z
# float64 confidence
# float64[4] xyxy

bool detected
float32 u  # Center x-coordinate in pixels
float32 v  # Center y-coordinate in pixels
float32 x  # 3D x-coordinate (meters)
float32 y  # 3D y-coordinate (meters)
float32 z  # 3D z-coordinate (meters)
float32 confidence  # Detection confidence
float32[] xyxy 
string robot_namespace  # Add this