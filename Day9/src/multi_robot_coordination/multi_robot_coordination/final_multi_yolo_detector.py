# # #!/usr/bin/env python3
# # import rclpy
# # from rclpy.node import Node
# # from std_msgs.msg import Bool, Header
# # from geometry_msgs.msg import PointStamped
# # from sensor_msgs.msg import Image, CameraInfo
# # from multi_robot_coordination.msg import CylinderDetection
# # import numpy as np
# # import cv2
# # import tkinter as tk
# # from PIL import Image as PILImage, ImageTk
# # from functools import partial
# # from threading import Thread
# # import screeninfo
# # import os
# # from datetime import datetime
# # from ultralytics import YOLO
# # import sys
# # import time

# # class TargetDetector(Node):
# #     def __init__(self):
# #         super().__init__('target_detector')
# #         self.cv_bridge_available = False
# #         try:
# #             from cv_bridge import CvBridge
# #             self.bridge = CvBridge()
# #             self.cv_bridge_available = True
# #             self.get_logger().info("cv_bridge imported successfully")
# #         except Exception as e:
# #             self.get_logger().error(f"cv_bridge import failed: {e}")
# #             self.get_logger().info("Will attempt manual image conversion")

# #         self.best_contour = None

# #         # Load YOLO model
# #         self.model_path = "/home/biswash/up_work/final_output/images/best.pt"
# #         try:
# #             self.model = YOLO(self.model_path)
# #             self.get_logger().info(f"Loaded YOLO model from {self.model_path}")
# #         except Exception as e:
# #             self.get_logger().error(f"Failed to load YOLO model: {e}")
# #             sys.exit(1)

# #         # Define robot configuration - 3 robots in a single row
# #         self.robot_namespaces = ['/tb1', '/tb2', '/tb3']
# #         self.num_robots = len(self.robot_namespaces)
# #         self.get_logger().info(f"Running TargetDetector for {self.num_robots} robots: {self.robot_namespaces}")

# #         # Save path parameter
# #         self.save_path = self.declare_parameter('save_path', './images').value
# #         try:
# #             os.makedirs(self.save_path, exist_ok=True)
# #             self.get_logger().info(f"Image save directory: {self.save_path}")
# #         except Exception as e:
# #             self.get_logger().error(f"Failed to create save directory {self.save_path}: {e}")
# #             self.save_path = None

# #         # Store camera info and latest images per robot
# #         self.camera_info = {}
# #         self.latest_images = {}
# #         self.depth_images = {}
# #         self.latest_detections = {}

# #         # GUI-related attributes
# #         self.status_labels = {}
# #         self.image_labels = {}
# #         self.detection_labels = {}
# #         self.images_visible = False
# #         self.imgtk_refs = {}

# #         self.camera_matrix_logged = {}

# #         # Shared publisher for global detections
# #         self.global_publisher = self.create_publisher(CylinderDetection, '/global_cylinder_detections', 10)
# #         self.get_logger().info("📢 Publisher created for /global_cylinder_detections")

# #         # Initialize GUI first
# #         self.root = None
# #         self.setup_gui()
# #         self.setup_ros_connections()

# #     def setup_gui(self):
# #         try:
# #             self.root = tk.Tk()
# #             self.root.title("Multi-Robot Target Detection")
            
# #             self.root.lift()
# #             self.root.attributes('-topmost', True)
# #             self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
# #             try:
# #                 screen = screeninfo.get_monitors()[0]
# #                 width = min(screen.width, 400 * self.num_robots)
# #                 height = screen.height // 2
# #                 x = (screen.width - width) // 2
# #                 y = (screen.height - height) // 2
# #                 self.root.geometry(f"{width}x{height}+{x}+{y}")
# #                 self.get_logger().info(f"GUI window size: {width}x{height} at position ({x}, {y})")
# #             except Exception as e:
# #                 self.get_logger().error(f"Screeninfo error: {e}. Using default window size.")
# #                 self.root.geometry("1200x600")  # Adjusted for 3 robots

# #             btn_frame = tk.Frame(self.root)
# #             btn_frame.pack(pady=5)
            
# #             self.toggle_btn = tk.Button(btn_frame, text="Show Images", command=self.toggle_images)
# #             self.toggle_btn.pack(side=tk.LEFT, padx=5)
            
# #             status_btn = tk.Button(btn_frame, text="GUI Status: Active", 
# #                                  command=lambda: self.get_logger().info("GUI is responsive!"))
# #             status_btn.pack(side=tk.LEFT, padx=5)

# #             self.robots_frame = tk.Frame(self.root)
# #             self.robots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# #             self.root.update_idletasks()
# #             self.root.update()
            
# #             self.get_logger().info("GUI setup completed successfully")
            
# #         except Exception as e:
# #             self.get_logger().error(f"Failed to setup GUI: {e}")
# #             raise

# #     def setup_ros_connections(self):
# #         self.detection_publishers = {}
# #         self.position_publishers = {}
# #         self.image_subscriptions = {}
# #         self.depth_subscriptions = {}
# #         self.camera_info_subscriptions = {}

# #         # Setup each robot by namespace
# #         for i, namespace in enumerate(self.robot_namespaces):
# #             self.setup_robot(namespace, i)

# #     def setup_robot(self, namespace, index):
# #         base_topic = f"{namespace}/intel_realsense_r200_depth"

# #         image_topic = f"{base_topic}/image_raw"
# #         depth_topic = f"{base_topic}/depth/image_raw"
# #         camera_info_topic = f"{base_topic}/depth/camera_info"

# #         self.image_subscriptions[namespace] = self.create_subscription(
# #             Image, image_topic, partial(self.image_callback, robot_ns=namespace), 10)
# #         self.depth_subscriptions[namespace] = self.create_subscription(
# #             Image, depth_topic, partial(self.depth_callback, robot_ns=namespace), 10)
# #         self.camera_info_subscriptions[namespace] = self.create_subscription(
# #             CameraInfo, camera_info_topic, partial(self.camera_info_callback, robot_ns=namespace), 10)

# #         self.detection_publishers[namespace] = self.create_publisher(Bool, f"{namespace}/target_detected", 10)
# #         self.position_publishers[namespace] = self.create_publisher(PointStamped, f"{namespace}/target_position", 10)

# #         self.latest_images[namespace] = None
# #         self.depth_images[namespace] = None
# #         self.latest_detections[namespace] = (False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0])
# #         self.imgtk_refs[namespace] = None

# #         self.create_robot_gui(namespace, index)

# #     def create_robot_gui(self, namespace, index):
# #         try:
# #             frame = tk.Frame(self.robots_frame, bd=2, relief=tk.RAISED, bg='lightgray')
            
# #             # Arrange robots in a single row (3 columns)
# #             frame.grid(row=0, column=index, padx=5, pady=5, sticky="nsew")
            
# #             # Configure grid weights for equal distribution
# #             self.robots_frame.grid_rowconfigure(0, weight=1)
# #             self.robots_frame.grid_columnconfigure(index, weight=1)

# #             label_title = tk.Label(frame, text=f"Robot {namespace}", 
# #                                  font=("Arial", 12, "bold"), bg='lightgray')
# #             label_title.pack(pady=2)

# #             status_label = tk.Label(frame, text="Status: Initializing", 
# #                                   font=("Arial", 10), bg='lightgray', fg='blue')
# #             status_label.pack(pady=2)
# #             self.status_labels[namespace] = status_label

# #             img_label = tk.Label(frame, bg='black')
# #             self.image_labels[namespace] = img_label

# #             detection_label = tk.Label(frame, text="Detection: None", 
# #                                      font=("Arial", 9), bg='lightgray')
# #             detection_label.pack(pady=2)
# #             self.detection_labels[namespace] = detection_label

# #             save_btn = tk.Button(frame, text="Save Image", 
# #                                command=partial(self.save_image, robot_ns=namespace),
# #                                bg='lightblue')
# #             save_btn.pack(pady=3)

# #             self.display_placeholder_image(namespace)
# #             self.get_logger().info(f"Created GUI for robot {namespace} at column {index}")
            
# #         except Exception as e:
# #             self.get_logger().error(f"Failed to create GUI for {namespace}: {e}")

# #     def display_placeholder_image(self, robot_ns):
# #         try:
# #             placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
# #             placeholder[:, :] = [64, 64, 64]
            
# #             text = f"Waiting for\n{robot_ns}"
# #             y_offset = 100
# #             for i, line in enumerate(text.split('\n')):
# #                 cv2.putText(placeholder, line, (50, y_offset + i*30),
# #                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
# #             placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
# #             img = PILImage.fromarray(placeholder)
# #             imgtk = ImageTk.PhotoImage(image=img)
            
# #             def safe_update():
# #                 try:
# #                     if robot_ns in self.image_labels:
# #                         self.image_labels[robot_ns].configure(image=imgtk)
# #                         self.image_labels[robot_ns].image = imgtk
# #                         self.imgtk_refs[robot_ns] = imgtk
# #                 except Exception as e:
# #                     self.get_logger().error(f"Failed to update placeholder for {robot_ns}: {e}")
            
# #             if self.root:
# #                 self.root.after(0, safe_update)
                
# #         except Exception as e:
# #             self.get_logger().error(f"[{robot_ns}] Placeholder image error: {e}")

# #     def save_image(self, robot_ns):
# #         if self.save_path is None or robot_ns not in self.latest_images or self.latest_images[robot_ns] is None:
# #             self.get_logger().error(f"[{robot_ns}] No image available to save")
# #             return
            
# #         try:
# #             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #             filename = os.path.join(self.save_path, f"{robot_ns.lstrip('/')}_{timestamp}.jpg")
# #             img_to_save = self.latest_images[robot_ns].copy()
            
# #             detected, u, v, confidence, xyxy = self.latest_detections[robot_ns]
# #             if detected:
# #                 x1, y1, x2, y2 = map(int, xyxy)
# #                 cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 cv2.putText(img_to_save, f"Cylinder ({confidence:.2f})", (x1, y1 - 10),
# #                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
# #             success = cv2.imwrite(filename, img_to_save)
# #             if success:
# #                 self.get_logger().info(f"[{robot_ns}] Saved image: {filename}")
# #             else:
# #                 self.get_logger().error(f"[{robot_ns}] Failed to write image: {filename}")
                
# #         except Exception as e:
# #             self.get_logger().error(f"[{robot_ns}] Failed to save image: {e}")

# #     def toggle_images(self):
# #         self.images_visible = not self.images_visible
# #         self.get_logger().info(f"Images visible: {self.images_visible}")
# #         self.toggle_btn.config(text="Hide Images" if self.images_visible else "Show Images")
        
# #         for ns, img_label in self.image_labels.items():
# #             try:
# #                 if self.images_visible:
# #                     img_label.pack(pady=5)
# #                     if self.imgtk_refs.get(ns) is None or self.latest_images.get(ns) is None:
# #                         self.display_placeholder_image(ns)
# #                 else:
# #                     img_label.pack_forget()
# #             except Exception as e:
# #                 self.get_logger().error(f"Error toggling image for {ns}: {e}")

# #     def camera_info_callback(self, msg, robot_ns):
# #         if robot_ns not in self.camera_info:
# #             self.camera_info[robot_ns] = msg
# #             if robot_ns not in self.camera_matrix_logged:
# #                 self.get_logger().info(f"[{robot_ns}] Camera matrix received: fx={msg.k[0]}, fy={msg.k[4]}, cx={msg.k[2]}, cy={msg.k[5]}")
# #                 self.camera_matrix_logged[robot_ns] = True
            
# #             def update_status():
# #                 if robot_ns in self.status_labels:
# #                     self.status_labels[robot_ns].config(text="Status: Camera Ready", fg='green')
            
# #             if self.root:
# #                 self.root.after(0, update_status)

# #     def depth_callback(self, msg, robot_ns):
# #         try:
# #             if msg.encoding == "32FC1":
# #                 depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
# #                 self.get_logger().info(f"[{robot_ns}] Manually converted depth image, shape: {depth_img.shape}")
# #             else:
# #                 self.get_logger().warning(f"[{robot_ns}] Unsupported depth encoding: {msg.encoding}, skipping depth")
# #                 return
# #             self.depth_images[robot_ns] = depth_img
# #             self.get_logger().info(f"[{robot_ns}] Depth image received, shape: {depth_img.shape}")
# #         except Exception as e:
# #             self.get_logger().error(f"[{robot_ns}] Manual depth image conversion failed: {e}")
# #             self.depth_images[robot_ns] = None

# #     def image_callback(self, msg, robot_ns):
# #         try:
# #             cv_image = self.manual_imgmsg_to_cv2(msg)
# #             if cv_image is None:
# #                 raise ValueError("Empty image")
# #             self.latest_images[robot_ns] = cv_image
            
# #             self.detect_and_publish(robot_ns, cv_image, msg)

# #         except Exception as e:
# #             self.get_logger().error(f"[{robot_ns}] Image callback error: {e}")

# #     def detect_and_publish(self, robot_ns, cv_image, img_msg):
# #         self.get_logger().info(f"📤 Processing detection for {robot_ns}")
# #         detection_msg = CylinderDetection()
# #         detection_msg.header = Header()
# #         detection_msg.header.stamp = self.get_clock().now().to_msg()
# #         detection_msg.detected = False
# #         detection_msg.u = 0.0
# #         detection_msg.v = 0.0
# #         detection_msg.x = 0.0
# #         detection_msg.y = 0.0
# #         detection_msg.z = 0.0
# #         detection_msg.confidence = 0.0
# #         detection_msg.x_min = 0.0
# #         detection_msg.y_min = 0.0
# #         detection_msg.x_max = 0.0
# #         detection_msg.y_max = 0.0
# #         detection_msg.image_width = float(img_msg.width)
# #         detection_msg.robot_namespace = robot_ns

# #         if cv_image is not None:
# #             display_image = cv_image.copy()
            
# #             detected, u, v, confidence, xyxy = self.detect_with_yolo(cv_image)
# #             self.get_logger().info(f"Detect cylinder result for {robot_ns}: detected={detected}, u={u}, v={v}, confidence={confidence}")

# #             if detected:
# #                 detection_msg.detected = True
# #                 detection_msg.u = float(u)
# #                 detection_msg.v = float(v)
# #                 detection_msg.confidence = float(confidence)
# #                 detection_msg.x_min = float(xyxy[0])
# #                 detection_msg.y_min = float(xyxy[1])
# #                 detection_msg.x_max = float(xyxy[2])
# #                 detection_msg.y_max = float(xyxy[3])

# #                 if robot_ns in self.camera_info and robot_ns in self.depth_images and self.depth_images[robot_ns] is not None:
# #                     depth_img = self.depth_images[robot_ns]
# #                     try:
# #                         height, width = depth_img.shape
# #                         if 0 <= u < width and 0 <= v < height:
# #                             depth = depth_img[v, u]
# #                             self.get_logger().info(f"[{robot_ns}] Raw depth value at (u={u}, v={v}): {depth} meters")
# #                             if depth > 0 and not np.isnan(depth):
# #                                 K = np.array(self.camera_info[robot_ns].k).reshape(3, 3)
# #                                 fx, fy = K[0, 0], K[1, 1]
# #                                 cx, cy = K[0, 2], K[1, 2]
# #                                 Z = float(depth)  # Depth is already in meters
# #                                 X = (u - cx) * Z / fx
# #                                 Y = (v - cy) * Z / fy
# #                                 detection_msg.x = float(X)
# #                                 detection_msg.y = float(Y)
# #                                 detection_msg.z = float(Z)
# #                                 self.get_logger().info(f"[{robot_ns}] Calculated 3D position: x={X:.3f}, y={Y:.3f}, z={Z:.3f} meters")
# #                             else:
# #                                 self.get_logger().warn(f"⚠️ Invalid depth value for {robot_ns}, publishing with zero coordinates")
# #                         else:
# #                             self.get_logger().warn(f"⚠️ Coordinates out of bounds for {robot_ns}, publishing with zero coordinates")
# #                     except Exception as e:
# #                         self.get_logger().error(f"[{robot_ns}] Depth processing error: {e}")
# #                         self.get_logger().warn(f"⚠️ 3D position calculation failed for {robot_ns}, publishing with zero coordinates")
                
# #                 x1, y1, x2, y2 = map(int, xyxy)
# #                 cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 label = f"Cylinder: {confidence:.2f}"
# #                 cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
# #                 self.latest_detections[robot_ns] = (detected, u, v, confidence, xyxy)

# #             self.latest_images[robot_ns] = display_image

# #         if detection_msg.detected:
# #             self.get_logger().info(f"📩 Publishing detection for {robot_ns}: detected={detection_msg.detected}, confidence={detection_msg.confidence}, x_min={detection_msg.x_min}, x_max={detection_msg.x_max}")
# #             self.global_publisher.publish(detection_msg)

# #         target_detected_msg = Bool()
# #         target_detected_msg.data = detected
# #         self.detection_publishers[robot_ns].publish(target_detected_msg)

# #         if detected:
# #             target_position_msg = PointStamped()
# #             target_position_msg.header.stamp = self.get_clock().now().to_msg()
# #             target_position_msg.header.frame_id = self.camera_info[robot_ns].header.frame_id if robot_ns in self.camera_info else robot_ns
# #             target_position_msg.point.x = float(detection_msg.x)
# #             target_position_msg.point.y = float(detection_msg.y)
# #             target_position_msg.point.z = float(detection_msg.z)
# #             self.position_publishers[robot_ns].publish(target_position_msg)

# #         self.update_gui(robot_ns, display_image, detected, u, v, confidence, xyxy, detection_msg.z)

# #     def detect_with_yolo(self, cv_image):
# #         try:
# #             results = self.model(cv_image, conf=0.2)
# #             if len(results) == 0 or len(results[0].boxes) == 0:
# #                 return False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0]
                
# #             best_idx = np.argmax(results[0].boxes.conf.cpu().numpy())
# #             best_box = results[0].boxes[best_idx]
# #             xyxy = best_box.xyxy[0].cpu().numpy()
            
# #             x_center = int((xyxy[0] + xyxy[2]) / 2)
# #             y_center = int((xyxy[1] + xyxy[3]) / 2)
# #             confidence = best_box.conf.item()
            
# #             return True, x_center, y_center, confidence, xyxy
            
# #         except Exception as e:
# #             self.get_logger().error(f"YOLO detection error: {e}")
# #             return False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0]

# #     def update_gui(self, robot_ns, cv_image, detected, u, v, confidence, xyxy, z):
# #         try:
# #             status_text = f"Status: {'Target Found!' if detected else 'Searching...'}"
# #             if detected:
# #                 status_text += f" ({confidence:.2f})"
                
# #             detection_text = f"Detection: {'YES' if detected else 'NO'}"
# #             if detected:
# #                 detection_text += f"\nPixel: ({u}, {v})\nDepth: {z:.3f} m"

# #             display_image = cv2.resize(cv_image, (320, 240))
# #             display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

# #             if detected:
# #                 orig_height, orig_width = cv_image.shape[:2]
# #                 scale_x = 320 / orig_width
# #                 scale_y = 240 / orig_height
# #                 x1, y1, x2, y2 = map(int, xyxy)
# #                 display_x1 = int(x1 * scale_x)
# #                 display_y1 = int(y1 * scale_y)
# #                 display_x2 = int(x2 * scale_x)
# #                 display_y2 = int(y2 * scale_y)
                
# #                 display_x1 = max(0, min(319, display_x1))
# #                 display_y1 = max(0, min(239, display_y1))
# #                 display_x2 = max(0, min(319, display_x2))
# #                 display_y2 = max(0, min(239, display_y2))
                
# #                 self.get_logger().info(f"[{robot_ns}] Drawing bounding box at ({display_x1}, {display_y1}) to ({display_x2}, {display_y2})")
# #                 cv2.rectangle(display_image, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 2)
# #                 cv2.putText(display_image, f"Cylinder {confidence:.2f}",
# #                           (display_x1 + 5, display_y1 - 5),
# #                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
# #                 cv2.putText(display_image, f"Depth: {z:.3f} m",
# #                           (display_x1 + 5, display_y2 + 15),
# #                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# #             img = PILImage.fromarray(display_image)
# #             imgtk = ImageTk.PhotoImage(image=img)

# #             def safe_gui_update():
# #                 try:
# #                     if robot_ns in self.status_labels:
# #                         self.status_labels[robot_ns].config(
# #                             text=status_text, 
# #                             fg='green' if detected else 'blue'
# #                         )
                    
# #                     if robot_ns in self.detection_labels:
# #                         self.detection_labels[robot_ns].config(text=detection_text)
                    
# #                     if self.images_visible and robot_ns in self.image_labels:
# #                         self.image_labels[robot_ns].configure(image=imgtk)
# #                         self.image_labels[robot_ns].image = imgtk
# #                         self.imgtk_refs[robot_ns] = imgtk
                        
# #                 except Exception as e:
# #                     self.get_logger().error(f"GUI update error for {robot_ns}: {e}")

# #             if self.root:
# #                 self.root.after(0, safe_gui_update)
                
# #         except Exception as e:
# #             self.get_logger().error(f"[{robot_ns}] GUI update failed: {e}")

# #     def manual_imgmsg_to_cv2(self, img_msg):
# #         try:
# #             self.get_logger().info(f"[{img_msg.header.frame_id}] Image encoding: {img_msg.encoding}, height: {img_msg.height}, width: {img_msg.width}")
            
# #             if img_msg.encoding == "bgr8":
# #                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
# #                 return img
# #             elif img_msg.encoding == "rgb8":
# #                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
# #                 return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# #             elif img_msg.encoding == "mono8":
# #                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
# #                 return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# #             elif img_msg.encoding == "rgba8":
# #                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 4)
# #                 return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
# #             else:
# #                 raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
                
# #         except Exception as e:
# #             self.get_logger().error(f"[{img_msg.header.frame_id}] Image conversion error: {e}")
# #             return None

# # def ros_spin(node):
# #     try:
# #         rclpy.spin(node)
# #     except Exception as e:
# #         node.get_logger().error(f"ROS spin error: {e}")
# #     finally:
# #         try:
# #             node.destroy_node()
# #             rclpy.shutdown()
# #         except:
# #             pass

# # def main(args=None):
# #     try:
# #         rclpy.init(args=args)
# #         node = TargetDetector()
        
# #         spin_thread = Thread(target=ros_spin, args=(node,), daemon=True)
# #         spin_thread.start()
        
# #         node.get_logger().info("Starting GUI main loop...")
# #         node.root.mainloop()
        
# #     except KeyboardInterrupt:
# #         print("\nShutting down...")
# #     except Exception as e:
# #         print(f"Error in main: {e}")
# #     finally:
# #         try:
# #             rclpy.shutdown()
# #         except:
# #             pass

# # if __name__ == '__main__':
# #     main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Bool, Header
# from geometry_msgs.msg import PointStamped
# from sensor_msgs.msg import Image, CameraInfo
# from multi_robot_coordination.msg import CylinderDetection
# import numpy as np
# import cv2
# import tkinter as tk
# from PIL import Image as PILImage, ImageTk
# from functools import partial
# from threading import Thread
# import screeninfo
# import os
# from datetime import datetime
# from ultralytics import YOLO
# import sys
# import time

# class TargetDetector(Node):
#     def __init__(self):
#         super().__init__('target_detector')
#         self.cv_bridge_available = False
#         try:
#             from cv_bridge import CvBridge
#             self.bridge = CvBridge()
#             self.cv_bridge_available = True
#             self.get_logger().info("cv_bridge imported successfully")
#         except Exception as e:
#             self.get_logger().error(f"cv_bridge import failed: {e}")
#             self.get_logger().info("Will attempt manual image conversion")

#         self.best_contour = None

#         # Load YOLO model
#         self.model_path = "/home/biswash/up_work/final_output/images/best.pt"
#         try:
#             self.model = YOLO(self.model_path)
#             self.get_logger().info(f"Loaded YOLO model from {self.model_path}")
#         except Exception as e:
#             self.get_logger().error(f"Failed to load YOLO model: {e}")
#             sys.exit(1)

#         # Define robot configuration - 3 robots in a single row
        
#         self.robot_namespaces = ['/tb1', '/tb2', '/tb3']
        
#         # self.robot_namespaces = ['/tb1', '/tb2']


#         self.num_robots = len(self.robot_namespaces)
#         self.get_logger().info(f"Running TargetDetector for {self.num_robots} robots: {self.robot_namespaces}")

#         # Save path parameter
#         self.save_path = self.declare_parameter('save_path', './images').value
#         try:
#             os.makedirs(self.save_path, exist_ok=True)
#             self.get_logger().info(f"Image save directory: {self.save_path}")
#         except Exception as e:
#             self.get_logger().error(f"Failed to create save directory {self.save_path}: {e}")
#             self.save_path = None

#         # Store camera info and latest images per robot
#         self.camera_info = {}
#         self.latest_images = {}
#         self.depth_images = {}
#         self.latest_detections = {}

#         # GUI-related attributes
#         self.status_labels = {}
#         self

#         self.image_labels = {}
#         self.detection_labels = {}
#         self.images_visible = False
#         self.imgtk_refs = {}

#         self.camera_matrix_logged = {}

#         # Shared publisher for global detections
#         self.global_publisher = self.create_publisher(CylinderDetection, '/global_cylinder_detections', 10)
#         self.get_logger().info("📢 Publisher created for /global_cylinderdetections")

#         # Initialize GUI first
#         self.root = None
#         self.setup_gui()
#         self.setup_ros_connections()

#     def setup_gui(self):
#         try:
#             self.root = tk.Tk()
#             self.root.title("Multi-Robot Target Detection")
            
#             self.root.lift()
#             self.root.attributes('-topmost', True)
#             self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
#             try:
#                 screen = screeninfo.get_monitors()[0]
#                 width = min(screen.width, 400 * self.num_robots)

#                 height = screen.height // 2
#                 x = (screen.width - width) // 2
#                 y = (screen.height - height) // 2
#                 self.root.geometry(f"{width}x{height}+{x}+{y}")
#                 self.get_logger().info(f"GUI window size: {width}x{height} at position ({x}, {y})")
#             except Exception as e:
#                 self.get_logger().error(f"Screeninfo error: {e}. Using default window size.")
#                 self.root.geometry("1200x600")  # Adjusted for 3 robots

#             btn_frame = tk.Frame(self.root)
#             btn_frame.pack(pady=5)
            
#             self.toggle_btn = tk.Button(btn_frame, text="Show Images", command=self.toggle_images)
#             self.toggle_btn.pack(side=tk.LEFT, padx=5)
            
#             status_btn = tk.Button(btn_frame, text="GUI Status: Active", 
#                                  command=lambda: self.get_logger().info("GUI is responsive!"))
#             status_btn.pack(side=tk.LEFT, padx=5)

#             self.robots_frame = tk.Frame(self.root)
#             self.robots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#             self.root.update_idletasks()
#             self.root.update()
            
#             self.get_logger().info("GUI setup completed successfully")
            
#         except Exception as e:
#             self.get_logger().error(f"Failed to setup GUI: {e}")
#             raise

#     def setup_ros_connections(self):
#         self.detection_publishers = {}
#         self.position_publishers = {}
#         self.image_subscriptions = {}
#         self.depth_subscriptions = {}
#         self.camera_info_subscriptions = {}

#         # Setup global subscriber
#         self.global_subscriber = self.create_subscription(
#             CylinderDetection,
#             '/global_cylinder_detections',
#             self.global_detection_callback,
#             10
#         )
#         self.get_logger().info("📥 Subscribed to /global_cylinder_detections")

#         # Setup each robot by namespace
#         for i, namespace in enumerate(self.robot_namespaces):
#             self.setup_robot(namespace, i)

#     def setup_robot(self, namespace, index):
#         base_topic = f"{namespace}/intel_realsense_r200_depth"

#         image_topic = f"{base_topic}/image_raw"
#         depth_topic = f"{base_topic}/depth/image_raw"
#         camera_info_topic = f"{base_topic}/depth/camera_info"

#         self.image_subscriptions[namespace] = self.create_subscription(
#             Image, image_topic, partial(self.image_callback, robot_ns=namespace), 10)
#         self.depth_subscriptions[namespace] = self.create_subscription(
#             Image, depth_topic, partial(self.depth_callback, robot_ns=namespace), 10)
#         self.camera_info_subscriptions[namespace] = self.create_subscription(
#             CameraInfo, camera_info_topic, partial(self.camera_info_callback, robot_ns=namespace), 10)

#         self.detection_publishers[namespace] = self.create_publisher(Bool, f"{namespace}/target_detected", 10)
#         self.position_publishers[namespace] = self.create_publisher(PointStamped, f"{namespace}/target_position", 10)

#         self.latest_images[namespace] = None
#         self.depth_images[namespace] = None
#         self.latest_detections[namespace] = (False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0])
#         self.imgtk_refs[namespace] = None

#         self.create_robot_gui(namespace, index)

#     def create_robot_gui(self, namespace, index):
#         try:
#             frame = tk.Frame(self.robots_frame, bd=2, relief=tk.RAISED, bg='lightgray')
            
#             # Arrange robots in a single row (3 columns)
#             frame.grid(row=0, column=index, padx=5, pady=5, sticky="nsew")
            
#             # Configure grid weights for equal distribution
#             self.robots_frame.grid_rowconfigure(0, weight=1)
#             self.robots_frame.grid_columnconfigure(index, weight=1)

#             label_title = tk.Label(frame, text=f"Robot {namespace}", 
#                                  font=("Arial", 12, "bold"), bg='lightgray')
#             label_title.pack(pady=2)

#             status_label = tk.Label(frame, text="Status: Initializing", 
#                                   font=("Arial", 10), bg='lightgray', fg='blue')
#             status_label.pack(pady=2)
#             self.status_labels[namespace] = status_label

#             img_label = tk.Label(frame, bg='black')
#             self.image_labels[namespace] = img_label

#             detection_label = tk.Label(frame, text="Detection: None", 
#                                      font=("Arial", 9), bg='lightgray')
#             detection_label.pack(pady=2)
#             self.detection_labels[namespace] = detection_label

#             save_btn = tk.Button(frame, text="Save Image", 
#                                command=partial(self.save_image, robot_ns=namespace),
#                                bg='lightblue')
#             save_btn.pack(pady=3)

#             self.display_placeholder_image(namespace)
#             self.get_logger().info(f"Created GUI for robot {namespace} at column {index}")
            
#         except Exception as e:
#             self.get_logger().error(f"Failed to create GUI for {namespace}: {e}")

#     def display_placeholder_image(self, robot_ns):
#         try:
#             placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
#             placeholder[:, :] = [64, 64, 64]
            
#             text = f"Waiting for\n{robot_ns}"
#             y_offset = 100
#             for i, line in enumerate(text.split('\n')):
#                 cv2.putText(placeholder, line, (50, y_offset + i*30),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
#             placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
#             img = PILImage.fromarray(placeholder)
#             imgtk = ImageTk.PhotoImage(image=img)
            
#             def safe_update():
#                 try:
#                     if robot_ns in self.image_labels:
#                         self.image_labels[robot_ns].configure(image=imgtk)
#                         self.image_labels[robot_ns].image = imgtk
#                         self.imgtk_refs[robot_ns] = imgtk
#                 except Exception as e:
#                     self.get_logger().error(f"Failed to update placeholder for {robot_ns}: {e}")
            
#             if self.root:
#                 self.root.after(0, safe_update)
                
#         except Exception as e:
#             self.get_logger().error(f"[{robot_ns}] Placeholder image error: {e}")

#     def save_image(self, robot_ns):
#         if self.save_path is None or robot_ns not in self.latest_images or self.latest_images[robot_ns] is None:
#             self.get_logger().error(f"[{robot_ns}] No image available to save")
#             return
            
#         try:
#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             filename = os.path.join(self.save_path, f"{robot_ns.lstrip('/')}_{timestamp}.jpg")
#             img_to_save = self.latest_images[robot_ns].copy()
            
#             detected, u, v, confidence, xyxy = self.latest_detections[robot_ns]
#             if detected:
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img_to_save, f"Cylinder ({confidence:.2f})", (x1, y1 - 10),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
#             success = cv2.imwrite(filename, img_to_save)
#             if success:
#                 self.get_logger().info(f"[{robot_ns}] Saved image: {filename}")
#             else:
#                 self.get_logger().error(f"[{robot_ns}] Failed to write image: {filename}")
                
#         except Exception as e:
#             self.get_logger().error(f"[{robot_ns}] Failed to save image: {e}")

#     def toggle_images(self):
#         self.images_visible = not self.images_visible
#         self.get_logger().info(f"Images visible: {self.images_visible}")
#         self.toggle_btn.config(text="Hide Images" if self.images_visible else "Show Images")
        
#         for ns, img_label in self.image_labels.items():
#             try:
#                 if self.images_visible:
#                     img_label.pack(pady=5)
#                     if self.imgtk_refs.get(ns) is None or self.latest_images.get(ns) is None:
#                         self.display_placeholder_image(ns)
#                 else:
#                     img_label.pack_forget()
#             except Exception as e:
#                 self.get_logger().error(f"Error toggling image for {ns}: {e}")

#     def camera_info_callback(self, msg, robot_ns):
#         if robot_ns not in self.camera_info:
#             self.camera_info[robot_ns] = msg
#             if robot_ns not in self.camera_matrix_logged:
#                 self.get_logger().info(f"[{robot_ns}] Camera matrix received: fx={msg.k[0]}, fy={msg.k[4]}, cx={msg.k[2]}, cy={msg.k[5]}")
#                 self.camera_matrix_logged[robot_ns] = True
            
#             def update_status():
#                 if robot_ns in self.status_labels:
#                     self.status_labels[robot_ns].config(text="Status: Camera Ready", fg='green')
            
#             if self.root:
#                 self.root.after(0, update_status)

#     def depth_callback(self, msg, robot_ns):
#         try:
#             if msg.encoding == "32FC1":
#                 depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
#                 self.get_logger().info(f"[{robot_ns}] Manually converted depth image, shape: {depth_img.shape}")
#             else:
#                 self.get_logger().warning(f"[{robot_ns}] Unsupported depth encoding: {msg.encoding}, skipping depth")
#                 return
#             self.depth_images[robot_ns] = depth_img
#             self.get_logger().info(f"[{robot_ns}] Depth image received, shape: {depth_img.shape}")
#         except Exception as e:
#             self.get_logger().error(f"[{robot_ns}] Manual depth image conversion failed: {e}")
#             self.depth_images[robot_ns] = None

#     def image_callback(self, msg, robot_ns):
#         try:
#             cv_image = self.manual_imgmsg_to_cv2(msg)
#             if cv_image is None:
#                 raise ValueError("Empty image")
#             self.latest_images[robot_ns] = cv_image
            
#             self.detect_and_publish(robot_ns, cv_image, msg)

#         except Exception as e:
#             self.get_logger().error(f"[{robot_ns}] Image callback error: {e}")

#     # def detect_and_publish(self, robot_ns, cv_image, img_msg):
#     #     self.get_logger().info(f"📤 Processing detection for {robot_ns}")
#     #     detection_msg = CylinderDetection()
#     #     detection_msg.header = Header()
#     #     detection_msg.header.stamp = self.get_clock().now().to_msg()
#     #     detection_msg.detected = False
#     #     detection_msg.u = 0.0
#     #     detection_msg.v = 0.0
#     #     detection_msg.x = 0.0
#     #     detection_msg.y = 0.0
#     #     detection_msg.z = 0.0
#     #     detection_msg.confidence = 0.0
#     #     detection_msg.x_min = 0.0
#     #     detection_msg.y_min = 0.0
#     #     detection_msg.x_max = 0.0
#     #     detection_msg.y_max = 0.0
#     #     detection_msg.image_width = float(img_msg.width)
#     #     detection_msg.robot_namespace = robot_ns

#     #     if cv_image is not None:
#     #         display_image = cv_image.copy()
            
#     #         detected, u, v, confidence, xyxy = self.detect_with_yolo(cv_image)
#     #         self.get_logger().info(f"Detect cylinder result for {robot_ns}: detected={detected}, u={u}, v={v}, confidence={confidence}")

#     #         if detected:
#     #             detection_msg.detected = True
#     #             detection_msg.u = float(u)
#     #             detection_msg.v = float(v)
#     #             detection_msg.confidence = float(confidence)
#     #             detection_msg.x_min = float(xyxy[0])
#     #             detection_msg.y_min = float(xyxy[1])
#     #             detection_msg.x_max = float(xyxy[2])
#     #             detection_msg.y_max = float(xyxy[3])

#     #             if robot_ns in self.camera_info and robot_ns in self.depth_images and self.depth_images[robot_ns] is not None:
#     #                 depth_img = self.depth_images[robot_ns]
#     #                 try:
#     #                     height, width = depth_img.shape
#     #                     if 0 <= u < width and 0 <= v < height:
#     #                         depth = depth_img[v, u]
#     #                         self.get_logger().info(f"[{robot_ns}] Raw depth value at (u={u}, v={v}): {depth} meters")
#     #                         if depth > 0 and not np.isnan(depth):
#     #                             K = np.array(self.camera_info[robot_ns].k).reshape(3, 3)
#     #                             fx, fy = K[0, 0], K[1, 1]
#     #                             cx, cy = K[0, 2], K[1, 2]
#     #                             Z = float(depth)  # Depth is already in meters
#     #                             X = (u - cx) * Z / fx
#     #                             Y = (v - cy) * Z / fy
#     #                             detection_msg.x = float(X)
#     #                             detection_msg.y = float(Y)
#     #                             detection_msg.z = float(Z)
#     #                             self.get_logger().info(f"[{robot_ns}] Calculated 3D position: x={X:.3f}, y={Y:.3f}, z={Z:.3f} meters")
#     #                         else:
#     #                             self.get_logger().warn(f"⚠️ Invalid depth value for {robot_ns}, publishing with zero coordinates")
#     #                     else:
#     #                         self.get_logger().warn(f"⚠️ Coordinates out of bounds for {robot_ns}, publishing with zero coordinates")
#     #                 except Exception as e:
#     #                     self.get_logger().error(f"[{robot_ns}] Depth processing error: {e}")
#     #                     self.get_logger().warn(f"⚠️ 3D position calculation failed for {robot_ns}, publishing with zero coordinates")
                
#     #             x1, y1, x2, y2 = map(int, xyxy)
#     #             cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     #             label = f"Cylinder: {confidence:.2f}"
#     #             cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
#     #             self.latest_detections[robot_ns] = (detected, u, v, confidence, xyxy)

#     #         self.latest_images[robot_ns] = display_image

#     #     if detection_msg.detected:
#     #         self.get_logger().info(f"📩 Publishing detection for {robot_ns}: detected={detection_msg.detected}, confidence={detection_msg.confidence}, x_min={detection_msg.x_min}, x_max={detection_msg.x_max}")
#     #         self.global_publisher.publish(detection_msg)

#     #     target_detected_msg = Bool()
#     #     target_detected_msg.data = detected
#     #     self.detection_publishers[robot_ns].publish(target_detected_msg)

#     #     if detected:
#     #         target_position_msg = PointStamped()
#     #         target_position_msg.header.stamp = self.get_clock().now().to_msg()
#     #         target_position_msg.header.frame_id = self.camera_info[robot_ns].header.frame_id if robot_ns in self.camera_info else robot_ns
#     #         target_position_msg.point.x = float(detection_msg.x)
#     #         target_position_msg.point.y = float(detection_msg.y)
#     #         target_position_msg.point.z = float(detection_msg.z)
#     #         self.position_publishers[robot_ns].publish(target_position_msg)

#     #         # Update GUI for the detecting robot
#     #         self.update_gui(robot_ns, display_image, detected, u, v, confidence, xyxy, detection_msg.z, detected_by=robot_ns)

#     def detect_and_publish(self, robot_ns, cv_image, img_msg):
#         self.get_logger().info(f"📤 Processing detection for {robot_ns}")
#         detection_msg = CylinderDetection()
#         detection_msg.header = Header()
#         detection_msg.header.stamp = self.get_clock().now().to_msg()
#         detection_msg.detected = False
#         detection_msg.u = 0.0
#         detection_msg.v = 0.0
#         detection_msg.x = 0.0
#         detection_msg.y = 0.0
#         detection_msg.z = 0.0
#         detection_msg.confidence = 0.0
#         detection_msg.x_min = 0.0
#         detection_msg.y_min = 0.0
#         detection_msg.x_max = 0.0
#         detection_msg.y_max = 0.0
#         detection_msg.image_width = float(img_msg.width)
#         detection_msg.robot_namespace = robot_ns

#         if cv_image is not None:
#             display_image = cv_image.copy()
            
#             detected, u, v, confidence, xyxy = self.detect_with_yolo(cv_image)
#             self.get_logger().info(f"Detect cylinder result for {robot_ns}: detected={detected}, u={u}, v={v}, confidence={confidence}")

#             if detected:
#                 detection_msg.detected = True
#                 detection_msg.u = float(u)
#                 detection_msg.v = float(v)
#                 detection_msg.confidence = float(confidence)
#                 detection_msg.x_min = float(xyxy[0])
#                 detection_msg.y_min = float(xyxy[1])
#                 detection_msg.x_max = float(xyxy[2])
#                 detection_msg.y_max = float(xyxy[3])

#                 if robot_ns in self.camera_info and robot_ns in self.depth_images and self.depth_images[robot_ns] is not None:
#                     depth_img = self.depth_images[robot_ns]
#                     try:
#                         height, width = depth_img.shape
#                         if 0 <= u < width and 0 <= v < height:
#                             depth = depth_img[v, u]
#                             self.get_logger().info(f"[{robot_ns}] Raw depth value at (u={u}, v={v}): {depth} meters")
#                             if depth > 0 and not np.isnan(depth):
#                                 K = np.array(self.camera_info[robot_ns].k).reshape(3, 3)
#                                 fx, fy = K[0, 0], K[1, 1]
#                                 cx, cy = K[0, 2], K[1, 2]
#                                 Z = float(depth)  # Depth is already in meters
#                                 X = (u - cx) * Z / fx
#                                 Y = (v - cy) * Z / fy
#                                 detection_msg.x = float(X)
#                                 detection_msg.y = float(Y)
#                                 detection_msg.z = float(Z)
#                                 self.get_logger().info(f"[{robot_ns}] Calculated 3D position: x={X:.3f}, y={Y:.3f}, z={Z:.3f} meters")
#                             else:
#                                 self.get_logger().warn(f"⚠️ Invalid depth value for {robot_ns}, publishing with zero coordinates")
#                         else:
#                             self.get_logger().warn(f"⚠️ Coordinates out of bounds for {robot_ns}, publishing with zero coordinates")
#                     except Exception as e:
#                         self.get_logger().error(f"[{robot_ns}] Depth processing error: {e}")
#                         self.get_logger().warn(f"⚠️ 3D position calculation failed for {robot_ns}, publishing with zero coordinates")
                
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label = f"Cylinder: {confidence:.2f}"
#                 cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
#                 self.latest_detections[robot_ns] = (detected, u, v, confidence, xyxy)

#             # Always update the GUI with the latest image, whether or not there's a detection
#             self.latest_images[robot_ns] = display_image
#             self.update_gui(robot_ns, display_image, detected, u, v, confidence, xyxy, detection_msg.z, detected_by=robot_ns)

#         if detection_msg.detected:
#             self.get_logger().info(f"📩 Publishing detection for {robot_ns}: detected={detection_msg.detected}, confidence={detection_msg.confidence}, x_min={detection_msg.x_min}, x_max={detection_msg.x_max}")
#             self.global_publisher.publish(detection_msg)

#         target_detected_msg = Bool()
#         target_detected_msg.data = detected
#         self.detection_publishers[robot_ns].publish(target_detected_msg)

#         if detected:
#             target_position_msg = PointStamped()
#             target_position_msg.header.stamp = self.get_clock().now().to_msg()
#             target_position_msg.header.frame_id = self.camera_info[robot_ns].header.frame_id if robot_ns in self.camera_info else robot_ns
#             target_position_msg.point.x = float(detection_msg.x)
#             target_position_msg.point.y = float(detection_msg.y)
#             target_position_msg.point.z = float(detection_msg.z)
#             self.position_publishers[robot_ns].publish(target_position_msg)

#     def global_detection_callback(self, msg):
#         robot_ns = msg.robot_namespace
#         detected = msg.detected
#         confidence = msg.confidence
#         u, v = int(msg.u), int(msg.v)
#         z = msg.z
#         xyxy = [msg.x_min, msg.y_min, msg.x_max, msg.y_max]

#         # Update GUI for all robots to reflect the detection
#         for ns in self.robot_namespaces:
#             # Skip updating the detecting robot if it already processed its own detection
#             if ns == robot_ns and self.latest_detections[ns][0]:
#                 continue

#             # Update the GUI with the detection from the specific robot
#             status_text = f"Status: Target Found by {robot_ns}! ({confidence:.2f})"
#             detection_text = f"Detection: YES by {robot_ns}\nPixel: ({u}, {v})\nDepth: {z:.3f} m"

#             # Use the latest image if available, or a placeholder
#             cv_image = self.latest_images.get(ns, None)
#             if cv_image is None:
#                 cv_image = np.zeros((240, 320, 3), dtype=np.uint8)
#                 cv_image[:, :] = [64, 64, 64]  # Gray placeholder
#                 cv2.putText(cv_image, f"No image from {ns}", (50, 120),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

#             self.update_gui(ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=robot_ns)

#         # Update the detecting robot's GUI with its own detection details
#         if robot_ns in self.robot_namespaces and self.latest_detections[robot_ns][0]:
#             cv_image = self.latest_images.get(robot_ns, None)
#             if cv_image is not None:
#                 self.update_gui(robot_ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=robot_ns)

#     def detect_with_yolo(self, cv_image):
#         try:
#             results = self.model(cv_image, conf=0.2)
#             if len(results) == 0 or len(results[0].boxes) == 0:
#                 return False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0]
                
#             best_idx = np.argmax(results[0].boxes.conf.cpu().numpy())
#             best_box = results[0].boxes[best_idx]
#             xyxy = best_box.xyxy[0].cpu().numpy()
            
#             x_center = int((xyxy[0] + xyxy[2]) / 2)
#             y_center = int((xyxy[1] + xyxy[3]) / 2)
#             confidence = best_box.conf.item()
            
#             return True, x_center, y_center, confidence, xyxy
            
#         except Exception as e:
#             self.get_logger().error(f"YOLO detection error: {e}")
#             return False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0]

#     # def update_gui(self, robot_ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=None):
#     #     try:
#     #         # Determine the detecting robot
#     #         detected_by = detected_by or robot_ns  # Use robot_ns if detected_by is not provided
#     #         status_text = f"Status: {'Target Found by ' + detected_by + '!' if detected else 'Searching...'}"
#     #         if detected:
#     #             status_text += f" ({confidence:.2f})"
                
#     #         detection_text = f"Detection: {'YES by ' + detected_by if detected else 'NO'}"
#     #         if detected:
#     #             detection_text += f"\nPixel: ({u}, {v})\nDepth: {z:.3f} m"

#     #         display_image = cv2.resize(cv_image, (320, 240))
#     #         display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

#     #         # Only draw the bounding box if this robot is the one that made the detection
#     #         if detected and robot_ns == detected_by:
#     #             orig_height, orig_width = cv_image.shape[:2]
#     #             scale_x = 320 / orig_width
#     #             scale_y = 240 / orig_height
#     #             x1, y1, x2, y2 = map(int, xyxy)
#     #             display_x1 = int(x1 * scale_x)
#     #             display_y1 = int(y1 * scale_y)
#     #             display_x2 = int(x2 * scale_x)
#     #             display_y2 = int(y2 * scale_y)
                
#     #             display_x1 = max(0, min(319, display_x1))
#     #             display_y1 = max(0, min(239, display_y1))
#     #             display_x2 = max(0, min(319, display_x2))
#     #             display_y2 = max(0, min(239, display_y2))
                
#     #             self.get_logger().info(f"[{robot_ns}] Drawing bounding box at ({display_x1}, {display_y1}) to ({display_x2}, {display_y2})")
#     #             cv2.rectangle(display_image, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 2)
#     #             cv2.putText(display_image, f"Cylinder {confidence:.2f} by {detected_by}",
#     #                         (display_x1 + 5, display_y1 - 5),
#     #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#     #             cv2.putText(display_image, f"Depth: {z:.3f} m",
#     #                         (display_x1 + 5, display_y2 + 15),
#     #                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

#     #         img = PILImage.fromarray(display_image)
#     #         imgtk = ImageTk.PhotoImage(image=img)

#     #         def safe_gui_update():
#     #             try:
#     #                 if robot_ns in self.status_labels:
#     #                     self.status_labels[robot_ns].config(
#     #                         text=status_text, 
#     #                         fg='green' if detected else 'blue'
#     #                     )
                    
#     #                 if robot_ns in self.detection_labels:
#     #                     self.detection_labels[robot_ns].config(text=detection_text)
                    
#     #                 if self.images_visible and robot_ns in self.image_labels:
#     #                     self.image_labels[robot_ns].configure(image=imgtk)
#     #                     self.image_labels[robot_ns].image = imgtk
#     #                     self.imgtk_refs[robot_ns] = imgtk
                        
#     #             except Exception as e:
#     #                 self.get_logger().error(f"GUI update error for {robot_ns}: {e}")

#     #         if self.root:
#     #             self.root.after(0, safe_gui_update)
                
#     #     except Exception as e:
#     #         self.get_logger().error(f"[{robot_ns}] GUI update failed: {e}")

#     def update_gui(self, robot_ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=None):
#         try:
#             # Determine the detecting robot
#             detected_by = detected_by or robot_ns  # Use robot_ns if detected_by is not provided
#             status_text = f"Status: {'Target Found by ' + detected_by + '!' if detected else 'Searching...'}"
#             if detected:
#                 status_text += f" ({confidence:.2f})"
                
#             detection_text = f"Detection: {'YES by ' + detected_by if detected else 'NO'}"
#             if detected:
#                 detection_text += f"\nPixel: ({u}, {v})\nDepth: {z:.3f} m"

#             display_image = cv2.resize(cv_image, (320, 240))
#             display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

#             # Only draw the bounding box if this robot is the one that made the detection
#             if detected and robot_ns == detected_by:
#                 orig_height, orig_width = cv_image.shape[:2]
#                 scale_x = 320 / orig_width
#                 scale_y = 240 / orig_height
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 display_x1 = int(x1 * scale_x)
#                 display_y1 = int(y1 * scale_y)
#                 display_x2 = int(x2 * scale_x)
#                 display_y2 = int(y2 * scale_y)
                
#                 display_x1 = max(0, min(319, display_x1))
#                 display_y1 = max(0, min(239, display_y1))
#                 display_x2 = max(0, min(319, display_x2))
#                 display_y2 = max(0, min(239, display_y2))
                
#                 self.get_logger().info(f"[{robot_ns}] Drawing bounding box at ({display_x1}, {display_y1}) to ({display_x2}, {display_y2})")
#                 cv2.rectangle(display_image, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 2)
#                 cv2.putText(display_image, f"Cylinder {confidence:.2f} by {detected_by}",
#                             (display_x1 + 5, display_y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#                 cv2.putText(display_image, f"Depth: {z:.3f} m",
#                             (display_x1 + 5, display_y2 + 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

#             img = PILImage.fromarray(display_image)
#             imgtk = ImageTk.PhotoImage(image=img)

#             def safe_gui_update():
#                 try:
#                     if robot_ns in self.status_labels:
#                         self.status_labels[robot_ns].config(
#                             text=status_text, 
#                             fg='green' if detected else 'blue'
#                         )
                    
#                     if robot_ns in self.detection_labels:
#                         self.detection_labels[robot_ns].config(text=detection_text)
                    
#                     if self.images_visible and robot_ns in self.image_labels:
#                         self.image_labels[robot_ns].configure(image=imgtk)
#                         self.image_labels[robot_ns].image = imgtk
#                         self.imgtk_refs[robot_ns] = imgtk
                        
#                 except Exception as e:
#                     self.get_logger().error(f"GUI update error for {robot_ns}: {e}")

#             if self.root:
#                 self.root.after(0, safe_gui_update)
                
#         except Exception as e:
#             self.get_logger().error(f"[{robot_ns}] GUI update failed: {e}")


#     def manual_imgmsg_to_cv2(self, img_msg):
#         try:
#             self.get_logger().info(f"[{img_msg.header.frame_id}] Image encoding: {img_msg.encoding}, height: {img_msg.height}, width: {img_msg.width}")
            
#             if img_msg.encoding == "bgr8":
#                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
#                 return img
#             elif img_msg.encoding == "rgb8":
#                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
#                 return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             elif img_msg.encoding == "mono8":
#                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
#                 return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             elif img_msg.encoding == "rgba8":
#                 img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 4)
#                 return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#             else:
#                 raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
                
#         except Exception as e:
#             self.get_logger().error(f"[{img_msg.header.frame_id}] Image conversion error: {e}")
#             return None

# def ros_spin(node):
#     try:
#         rclpy.spin(node)
#     except Exception as e:
#         node.get_logger().error(f"ROS spin error: {e}")
#     finally:
#         try:
#             node.destroy_node()
#             rclpy.shutdown()
#         except:
#             pass

# def main(args=None):
#     try:
#         rclpy.init(args=args)
#         node = TargetDetector()
        
#         spin_thread = Thread(target=ros_spin, args=(node,), daemon=True)
#         spin_thread.start()
        
#         node.get_logger().info("Starting GUI main loop...")
#         node.root.mainloop()
        
#     except KeyboardInterrupt:
#         print("\nShutting down...")
#     except Exception as e:
#         print(f"Error in main: {e}")
#     finally:
#         try:
#             rclpy.shutdown()
#         except:
#             pass

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from multi_robot_coordination.msg import CylinderDetection
import numpy as np
import cv2
import tkinter as tk
from PIL import Image as PILImage, ImageTk
from functools import partial
from threading import Thread
import screeninfo
import os
from datetime import datetime
from ultralytics import YOLO
import sys
import time

class TargetDetector(Node):
    def __init__(self):
        super().__init__('target_detector')
        self.cv_bridge_available = False
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.cv_bridge_available = True
            self.get_logger().info("cv_bridge imported successfully")
        except Exception as e:
            self.get_logger().error(f"cv_bridge import failed: {e}")
            self.get_logger().info("Will attempt manual image conversion")

        self.best_contour = None

        # Load YOLO model
        self.model_path = "/home/biswash/up_work/final_output/images/best.pt"
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            sys.exit(1)

        # Define robot configuration - 3 robots in a single row
        self.robot_namespaces = ['/tb1', '/tb2', '/tb3']
        self.num_robots = len(self.robot_namespaces)
        self.get_logger().info(f"Running TargetDetector for {self.num_robots} robots: {self.robot_namespaces}")

        # Save path parameter
        self.save_path = self.declare_parameter('save_path', './images').value
        try:
            os.makedirs(self.save_path, exist_ok=True)
            self.get_logger().info(f"Image save directory: {self.save_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to create save directory {self.save_path}: {e}")
            self.save_path = None

        # Store camera info, latest images, and detections per robot
        self.camera_info = {}
        self.latest_images = {}
        self.depth_images = {}
        self.latest_detections = {}

        # Store latest global pose
        self.latest_global_pose = None

        # GUI-related attributes
        self.status_labels = {}
        self.image_labels = {}
        self.detection_labels = {}
        self.images_visible = False
        self.imgtk_refs = {}
        self.camera_matrix_logged = {}

        # Shared publisher for global detections
        self.global_publisher = self.create_publisher(CylinderDetection, '/global_cylinder_detections', 10)
        self.get_logger().info("📢 Publisher created for /global_cylinder_detections")

        # Initialize GUI and ROS connections
        self.root = None
        self.setup_gui()
        self.setup_ros_connections()

    def setup_gui(self):
        try:
            self.root = tk.Tk()
            self.root.title("Multi-Robot Target Detection")
            
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            
            try:
                screen = screeninfo.get_monitors()[0]
                width = min(screen.width, 400 * self.num_robots)
                height = screen.height // 2
                x = (screen.width - width) // 2
                y = (screen.height - height) // 2
                self.root.geometry(f"{width}x{height}+{x}+{y}")
                self.get_logger().info(f"GUI window size: {width}x{height} at position ({x}, {y})")
            except Exception as e:
                self.get_logger().error(f"Screeninfo error: {e}. Using default window size.")
                self.root.geometry("1200x600")

            btn_frame = tk.Frame(self.root)
            btn_frame.pack(pady=5)
            
            self.toggle_btn = tk.Button(btn_frame, text="Show Images", command=self.toggle_images)
            self.toggle_btn.pack(side=tk.LEFT, padx=5)
            
            status_btn = tk.Button(btn_frame, text="GUI Status: Active", 
                                 command=lambda: self.get_logger().info("GUI is responsive!"))
            status_btn.pack(side=tk.LEFT, padx=5)

            self.robots_frame = tk.Frame(self.root)
            self.robots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.root.update_idletasks()
            self.root.update()
            
            self.get_logger().info("GUI setup completed successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup GUI: {e}")
            raise

    def setup_ros_connections(self):
        self.detection_publishers = {}
        self.position_publishers = {}
        self.image_subscriptions = {}
        self.depth_subscriptions = {}
        self.camera_info_subscriptions = {}

        # Setup global subscriber for cylinder detections
        self.global_subscriber = self.create_subscription(
            CylinderDetection,
            '/global_cylinder_detections',
            self.global_detection_callback,
            10
        )
        self.get_logger().info("📥 Subscribed to /global_cylinder_detections")

        # Setup subscriber for target world pose
        self.global_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/target_world_pose',
            self.global_pose_callback,
            10
        )
        self.get_logger().info("📥 Subscribed to /target_world_pose")

        # Setup each robot by namespace
        for i, namespace in enumerate(self.robot_namespaces):
            self.setup_robot(namespace, i)

    def setup_robot(self, namespace, index):
        base_topic = f"{namespace}/intel_realsense_r200_depth"

        image_topic = f"{base_topic}/image_raw"
        depth_topic = f"{base_topic}/depth/image_raw"
        camera_info_topic = f"{base_topic}/depth/camera_info"

        self.image_subscriptions[namespace] = self.create_subscription(
            Image, image_topic, partial(self.image_callback, robot_ns=namespace), 10)
        self.depth_subscriptions[namespace] = self.create_subscription(
            Image, depth_topic, partial(self.depth_callback, robot_ns=namespace), 10)
        self.camera_info_subscriptions[namespace] = self.create_subscription(
            CameraInfo, camera_info_topic, partial(self.camera_info_callback, robot_ns=namespace), 10)

        self.detection_publishers[namespace] = self.create_publisher(Bool, f"{namespace}/target_detected", 10)
        self.position_publishers[namespace] = self.create_publisher(PointStamped, f"{namespace}/target_position", 10)

        self.latest_images[namespace] = None
        self.depth_images[namespace] = None
        self.latest_detections[namespace] = (False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0])
        self.imgtk_refs[namespace] = None

        self.create_robot_gui(namespace, index)

    def create_robot_gui(self, namespace, index):
        try:
            frame = tk.Frame(self.robots_frame, bd=2, relief=tk.RAISED, bg='lightgray')
            frame.grid(row=0, column=index, padx=5, pady=5, sticky="nsew")
            
            self.robots_frame.grid_rowconfigure(0, weight=1)
            self.robots_frame.grid_columnconfigure(index, weight=1)

            label_title = tk.Label(frame, text=f"Robot {namespace}", 
                                 font=("Arial", 12, "bold"), bg='lightgray')
            label_title.pack(pady=2)

            status_label = tk.Label(frame, text="Status: Initializing", 
                                  font=("Arial", 10), bg='lightgray', fg='blue')
            status_label.pack(pady=2)
            self.status_labels[namespace] = status_label

            img_label = tk.Label(frame, bg='black')
            self.image_labels[namespace] = img_label

            detection_label = tk.Label(frame, text="Detection: None", 
                                     font=("Arial", 9), bg='lightgray')
            detection_label.pack(pady=2)
            self.detection_labels[namespace] = detection_label

            save_btn = tk.Button(frame, text="Save Image", 
                               command=partial(self.save_image, robot_ns=namespace),
                               bg='lightblue')
            save_btn.pack(pady=3)

            self.display_placeholder_image(namespace)
            self.get_logger().info(f"Created GUI for robot {namespace} at column {index}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to create GUI for {namespace}: {e}")

    def display_placeholder_image(self, robot_ns):
        try:
            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
            placeholder[:, :] = [64, 64, 64]
            
            text = f"Waiting for\n{robot_ns}"
            y_offset = 100
            for i, line in enumerate(text.split('\n')):
                cv2.putText(placeholder, line, (50, y_offset + i*30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            placeholder = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            img = PILImage.fromarray(placeholder)
            imgtk = ImageTk.PhotoImage(image=img)
            
            def safe_update():
                try:
                    if robot_ns in self.image_labels:
                        self.image_labels[robot_ns].configure(image=imgtk)
                        self.image_labels[robot_ns].image = imgtk
                        self.imgtk_refs[robot_ns] = imgtk
                except Exception as e:
                    self.get_logger().error(f"Failed to update placeholder for {robot_ns}: {e}")
            
            if self.root:
                self.root.after(0, safe_update)
                
        except Exception as e:
            self.get_logger().error(f"[{robot_ns}] Placeholder image error: {e}")

    def save_image(self, robot_ns):
        if self.save_path is None or robot_ns not in self.latest_images or self.latest_images[robot_ns] is None:
            self.get_logger().error(f"[{robot_ns}] No image available to save")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.save_path, f"{robot_ns.lstrip('/')}_{timestamp}.jpg")
            img_to_save = self.latest_images[robot_ns].copy()
            
            detected, u, v, confidence, xyxy = self.latest_detections[robot_ns]
            if detected:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_to_save, f"Cylinder ({confidence:.2f})", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            success = cv2.imwrite(filename, img_to_save)
            if success:
                self.get_logger().info(f"[{robot_ns}] Saved image: {filename}")
            else:
                self.get_logger().error(f"[{robot_ns}] Failed to write image: {filename}")
                
        except Exception as e:
            self.get_logger().error(f"[{robot_ns}] Failed to save image: {e}")

    def toggle_images(self):
        self.images_visible = not self.images_visible
        self.get_logger().info(f"Images visible: {self.images_visible}")
        self.toggle_btn.config(text="Hide Images" if self.images_visible else "Show Images")
        
        for ns, img_label in self.image_labels.items():
            try:
                if self.images_visible:
                    img_label.pack(pady=5)
                    if self.imgtk_refs.get(ns) is None or self.latest_images.get(ns) is None:
                        self.display_placeholder_image(ns)
                else:
                    img_label.pack_forget()
            except Exception as e:
                self.get_logger().error(f"Error toggling image for {ns}: {e}")

    def camera_info_callback(self, msg, robot_ns):
        if robot_ns not in self.camera_info:
            self.camera_info[robot_ns] = msg
            if robot_ns not in self.camera_matrix_logged:
                self.get_logger().info(f"[{robot_ns}] Camera matrix received: fx={msg.k[0]}, fy={msg.k[4]}, cx={msg.k[2]}, cy={msg.k[5]}")
                self.camera_matrix_logged[robot_ns] = True
            
            def update_status():
                if robot_ns in self.status_labels:
                    self.status_labels[robot_ns].config(text="Status: Camera Ready", fg='green')
            
            if self.root:
                self.root.after(0, update_status)

    def depth_callback(self, msg, robot_ns):
        try:
            if msg.encoding == "32FC1":
                depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                self.get_logger().info(f"[{robot_ns}] Manually converted depth image, shape: {depth_img.shape}")
            else:
                self.get_logger().warning(f"[{robot_ns}] Unsupported depth encoding: {msg.encoding}, skipping depth")
                return
            self.depth_images[robot_ns] = depth_img
            self.get_logger().info(f"[{robot_ns}] Depth image received, shape: {depth_img.shape}")
        except Exception as e:
            self.get_logger().error(f"[{robot_ns}] Manual depth image conversion failed: {e}")
            self.depth_images[robot_ns] = None

    def image_callback(self, msg, robot_ns):
        try:
            cv_image = self.manual_imgmsg_to_cv2(msg)
            if cv_image is None:
                raise ValueError("Empty image")
            self.latest_images[robot_ns] = cv_image
            
            self.detect_and_publish(robot_ns, cv_image, msg)

        except Exception as e:
            self.get_logger().error(f"[{robot_ns}] Image callback error: {e}")

    def detect_and_publish(self, robot_ns, cv_image, img_msg):
        self.get_logger().info(f"📤 Processing detection for {robot_ns}")
        detection_msg = CylinderDetection()
        detection_msg.header = Header()
        detection_msg.header.stamp = self.get_clock().now().to_msg()
        detection_msg.detected = False
        detection_msg.u = 0.0
        detection_msg.v = 0.0
        detection_msg.x = 0.0
        detection_msg.y = 0.0
        detection_msg.z = 0.0
        detection_msg.confidence = 0.0
        detection_msg.x_min = 0.0
        detection_msg.y_min = 0.0
        detection_msg.x_max = 0.0
        detection_msg.y_max = 0.0
        detection_msg.image_width = float(img_msg.width)
        detection_msg.robot_namespace = robot_ns

        if cv_image is not None:
            display_image = cv_image.copy()
            
            detected, u, v, confidence, xyxy = self.detect_with_yolo(cv_image)
            self.get_logger().info(f"Detect cylinder result for {robot_ns}: detected={detected}, u={u}, v={v}, confidence={confidence}")

            if detected:
                detection_msg.detected = True
                detection_msg.u = float(u)
                detection_msg.v = float(v)
                detection_msg.confidence = float(confidence)
                detection_msg.x_min = float(xyxy[0])
                detection_msg.y_min = float(xyxy[1])
                detection_msg.x_max = float(xyxy[2])
                detection_msg.y_max = float(xyxy[3])

                if robot_ns in self.camera_info and robot_ns in self.depth_images and self.depth_images[robot_ns] is not None:
                    depth_img = self.depth_images[robot_ns]
                    try:
                        height, width = depth_img.shape
                        if 0 <= u < width and 0 <= v < height:
                            depth = depth_img[v, u]
                            self.get_logger().info(f"[{robot_ns}] Raw depth value at (u={u}, v={v}): {depth} meters")
                            if depth > 0 and not np.isnan(depth):
                                K = np.array(self.camera_info[robot_ns].k).reshape(3, 3)
                                fx, fy = K[0, 0], K[1, 1]
                                cx, cy = K[0, 2], K[1, 2]
                                Z = float(depth)
                                X = (u - cx) * Z / fx
                                Y = (v - cy) * Z / fy
                                detection_msg.x = float(X)
                                detection_msg.y = float(Y)
                                detection_msg.z = float(Z)
                                self.get_logger().info(f"[{robot_ns}] Calculated 3D position: x={X:.3f}, y={Y:.3f}, z={Z:.3f} meters")
                            else:
                                self.get_logger().warn(f"⚠️ Invalid depth value for {robot_ns}, publishing with zero coordinates")
                        else:
                            self.get_logger().warn(f"⚠️ Coordinates out of bounds for {robot_ns}, publishing with zero coordinates")
                    except Exception as e:
                        self.get_logger().error(f"[{robot_ns}] Depth processing error: {e}")
                        self.get_logger().warn(f"⚠️ 3D position calculation failed for {robot_ns}, publishing with zero coordinates")
                
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Cylinder: {confidence:.2f}"
                cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                self.latest_detections[robot_ns] = (detected, u, v, confidence, xyxy)

            self.latest_images[robot_ns] = display_image
            self.update_gui(robot_ns, display_image, detected, u, v, confidence, xyxy, detection_msg.z, detected_by=robot_ns)

        if detection_msg.detected:
            self.get_logger().info(f"📩 Publishing detection for {robot_ns}: detected={detection_msg.detected}, confidence={detection_msg.confidence}, x_min={detection_msg.x_min}, x_max={detection_msg.x_max}")
            self.global_publisher.publish(detection_msg)

        target_detected_msg = Bool()
        target_detected_msg.data = detected
        self.detection_publishers[robot_ns].publish(target_detected_msg)

        if detected:
            target_position_msg = PointStamped()
            target_position_msg.header.stamp = self.get_clock().now().to_msg()
            target_position_msg.header.frame_id = self.camera_info[robot_ns].header.frame_id if robot_ns in self.camera_info else robot_ns
            target_position_msg.point.x = float(detection_msg.x)
            target_position_msg.point.y = float(detection_msg.y)
            target_position_msg.point.z = float(detection_msg.z)
            self.position_publishers[robot_ns].publish(target_position_msg)

    def global_pose_callback(self, msg):
        try:
            self.latest_global_pose = msg
            self.get_logger().info(
                f"Received /target_world_pose: x={msg.pose.position.x:.3f}, "
                f"y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}"
            )
        except Exception as e:
            self.get_logger().error(f"Error in global_pose_callback: {e}")

    def global_detection_callback(self, msg):
        robot_ns = msg.robot_namespace
        detected = msg.detected
        confidence = msg.confidence
        u, v = int(msg.u), int(msg.v)
        z = msg.z
        xyxy = [msg.x_min, msg.y_min, msg.x_max, msg.y_max]

        # Update GUI for all robots
        for ns in self.robot_namespaces:
            # Skip updating the detecting robot if it already processed its own detection
            if ns == robot_ns and self.latest_detections[ns][0]:
                continue

            # Use the latest image if available, or a placeholder
            cv_image = self.latest_images.get(ns, None)
            if cv_image is None:
                cv_image = np.zeros((240, 320, 3), dtype=np.uint8)
                cv_image[:, :] = [64, 64, 64]
                cv2.putText(cv_image, f"No image from {ns}", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            self.update_gui(ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=robot_ns)

        # Update the detecting robot's GUI
        if robot_ns in self.robot_namespaces and self.latest_detections[robot_ns][0]:
            cv_image = self.latest_images.get(robot_ns, None)
            if cv_image is not None:
                self.update_gui(robot_ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=robot_ns)

    def detect_with_yolo(self, cv_image):
        try:
            results = self.model(cv_image, conf=0.2)
            if len(results) == 0 or len(results[0].boxes) == 0:
                return False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0]
                
            best_idx = np.argmax(results[0].boxes.conf.cpu().numpy())
            best_box = results[0].boxes[best_idx]
            xyxy = best_box.xyxy[0].cpu().numpy()
            
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            confidence = best_box.conf.item()
            
            return True, x_center, y_center, confidence, xyxy
            
        except Exception as e:
            self.get_logger().error(f"YOLO detection error: {e}")
            return False, 0, 0, 0.0, [0.0, 0.0, 0.0, 0.0]

    def update_gui(self, robot_ns, cv_image, detected, u, v, confidence, xyxy, z, detected_by=None):
        try:
            detected_by = detected_by or robot_ns
            status_text = f"Status: {'Target Found by ' + detected_by + '!' if detected else 'No Detection'}"
            if detected:
                status_text += f" ({confidence:.2f})"
                
            detection_text = f"Detection: {'YES by ' + detected_by if detected else 'NO'}"
            if detected:
                detection_text += f"\nPixel: ({u}, {v})\nDepth: {z:.3f} m"
                if self.latest_global_pose is not None:
                    detection_text += f"\nGlobal: ({self.latest_global_pose.pose.position.x:.3f}, {self.latest_global_pose.pose.position.y:.3f})"

            display_image = cv2.resize(cv_image, (320, 240))
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

            if detected and robot_ns == detected_by:
                orig_height, orig_width = cv_image.shape[:2]
                scale_x = 320 / orig_width
                scale_y = 240 / orig_height
                x1, y1, x2, y2 = map(int, xyxy)
                display_x1 = int(x1 * scale_x)
                display_y1 = int(y1 * scale_y)
                display_x2 = int(x2 * scale_x)
                display_y2 = int(y2 * scale_y)
                
                display_x1 = max(0, min(319, display_x1))
                display_y1 = max(0, min(239, display_y1))
                display_x2 = max(0, min(319, display_x2))
                display_y2 = max(0, min(239, display_y2))
                
                self.get_logger().info(f"[{robot_ns}] Drawing bounding box at ({display_x1}, {display_y1}) to ({display_x2}, {display_y2})")
                cv2.rectangle(display_image, (display_x1, display_y1), (display_x2, display_y2), (0, 255, 0), 2)
                cv2.putText(display_image, f"Cylinder {confidence:.2f} by {detected_by}",
                            (display_x1 + 5, display_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(display_image, f"Depth: {z:.3f} m",
                            (display_x1 + 5, display_y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                if self.latest_global_pose is not None:
                    cv2.putText(display_image, f"Global: ({self.latest_global_pose.pose.position.x:.3f}, {self.latest_global_pose.pose.position.y:.3f})",
                                (display_x1 + 5, display_y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            img = PILImage.fromarray(display_image)
            imgtk = ImageTk.PhotoImage(image=img)

            def safe_gui_update():
                try:
                    if robot_ns in self.status_labels:
                        self.status_labels[robot_ns].config(
                            text=status_text, 
                            fg='green' if detected else 'blue'
                        )
                    
                    if robot_ns in self.detection_labels:
                        self.detection_labels[robot_ns].config(text=detection_text)
                    
                    if self.images_visible and robot_ns in self.image_labels:
                        self.image_labels[robot_ns].configure(image=imgtk)
                        self.image_labels[robot_ns].image = imgtk
                        self.imgtk_refs[robot_ns] = imgtk
                        
                except Exception as e:
                    self.get_logger().error(f"GUI update error for {robot_ns}: {e}")

            if self.root:
                self.root.after(0, safe_gui_update)
                
        except Exception as e:
            self.get_logger().error(f"[{robot_ns}] GUI update failed: {e}")

    def manual_imgmsg_to_cv2(self, img_msg):
        try:
            self.get_logger().info(f"[{img_msg.header.frame_id}] Image encoding: {img_msg.encoding}, height: {img_msg.height}, width: {img_msg.width}")
            
            if img_msg.encoding == "bgr8":
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
                return img
            elif img_msg.encoding == "rgb8":
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img_msg.encoding == "mono8":
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img_msg.encoding == "rgba8":
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 4)
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
                
        except Exception as e:
            self.get_logger().error(f"[{img_msg.header.frame_id}] Image conversion error: {e}")
            return None

def ros_spin(node):
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"ROS spin error: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass

def main(args=None):
    try:
        rclpy.init(args=args)
        node = TargetDetector()
        
        spin_thread = Thread(target=ros_spin, args=(node,), daemon=True)
        spin_thread.start()
        
        node.get_logger().info("Starting GUI main loop...")
        node.root.mainloop()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()