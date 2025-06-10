# 🚀 60 Days of Learning 2025 – Day 9  
## 🧠 Multi-Robot Auto Target Detection Using Navigation2  

Today, I worked on building a **multi-robot system** capable of **auto-target detection**. The goal was to have the robot **closest to the target** perform the detection task autonomously, leveraging the **ROS 2 Navigation2 framework**.  

📹 **Demo:** [Google Drive Link – Demo](#) *(https://drive.google.com/drive/folders/17UAKry4LWt2WpI-sqkbIQJmnDl1lEOQR?usp=sharing)*  

---

## 🛠️ Setup Instructions

### ✅ Prerequisite: Build the Workspace  
```bash
colcon build
source install/setup.bash
```

---

## 🧭 Step-by-Step Launch Instructions  

> ⚠️ **Note:** Source the setup file (`source install/setup.bash`) in **each terminal** before running the scripts.

### 1. Launch the Simulation Environment  
```bash
ros2 launch turtlebot3_gazebo gazebo_multi_maze_nav2_world.launch.py
```

### 2. Run the Target Detection & Coordination Scripts  

**Terminal 1:**
```bash
python3 src/multi_robot_coordination/multi_robot_coordination/final_multi_yolo_detector.py
```

**Terminal 2:**
```bash
python3 src/multi_robot_coordination/multi_robot_coordination/world_coordinate_converter.py
```

**Terminal 3:**
```bash
python3 src/multi_robot_coordination/multi_robot_coordination/split_1.py
```

**Terminal 4:**
```bash
python3 src/multi_robot_coordination/multi_robot_coordination/final_closest_robot.py
```

---

### 3. Launch Final World for Multi-Robot Navigation  
```bash
ros2 launch turtlebot3_multi_robot gazebo_multi_maze_nav2_world.launch.py
```

---

## 📌 Description  

This project enables a team of robots in simulation to:  
- Detect a target using a shared YOLO detector.  
- Convert detections to world coordinates.  
- Identify the closest robot to the target.  
- Send a navigation goal to the selected robot using Navigation2.
