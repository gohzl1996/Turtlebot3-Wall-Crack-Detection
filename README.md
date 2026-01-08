# 🤖 TurtleBot3 Wall Crack Detection  
**Autonomous Mobile Robot for Real-Time Infrastructure Inspection**

---

## 📌 Project Overview

This project implements an **autonomous TurtleBot3-based inspection system** capable of navigating indoor environments and **detecting wall surface cracks in real time** using a **CNN-based computer vision model**.

The system integrates **ROS 2 Humble**, **SLAM-based navigation**, **OpenCV**, and **deep learning inference** to demonstrate how mobile robots can assist in **automated infrastructure inspection**, reducing the need for manual and potentially hazardous inspections.

---

## 🎯 Objectives

- Enable **autonomous navigation** using SLAM and obstacle avoidance  
- Perform **real-time wall crack detection** using a trained CNN model  
- Overlay detection results on live camera feed  
- Visualize robot state and sensor data in **RViz**  
- Demonstrate a scalable inspection pipeline suitable for tunnels, corridors, and buildings  

---

## 🧠 System Architecture

### Crack Detection Pipeline
```
Camera Feed
   ↓
Image Preprocessing (OpenCV)
   ↓
CNN Crack Detection Model
   ↓
Crack / No-Crack Prediction
   ↓
Visualization Overlay
   ↓
ROS 2 Topics → RViz / Display
```

### Navigation Stack
```
LiDAR → SLAM → Localization → Path Planning → Motor Control
```

---

## 🧩 Key Features

- ✅ Autonomous navigation using **ROS 2 Navigation Stack**
- ✅ Real-time crack detection using **CNN**
- ✅ Onboard inference using **Python**
- ✅ Live visualization in **RViz**
- ✅ Modular **ROS 2 node-based architecture**
- ✅ Designed with real-world inspection use cases in mind

---

## 🛠️ Technologies Used

| Category | Tools |
|--------|------|
| Robot Platform | TurtleBot3 |
| Middleware | ROS 2 Humble |
| Navigation | SLAM, Nav2 |
| Computer Vision | OpenCV |
| Deep Learning | CNN (TensorFlow / Keras) |
| Programming | Python |
| Visualization | RViz |

---

## 🚀 How It Works

### 1️⃣ Autonomous Navigation
- TurtleBot3 uses **LiDAR-based SLAM** to build and update a map
- Navigation stack handles localization, path planning, and obstacle avoidance
- Robot autonomously traverses corridors and wall-adjacent paths

### 2️⃣ Image Acquisition
- Camera stream is subscribed via a ROS 2 image topic
- Frames are resized and normalized for neural network inference

### 3️⃣ Crack Detection
- A trained **CNN model** predicts the presence of wall cracks
- Prediction confidence is computed per frame
- Detection results are overlaid on the live camera feed

### 4️⃣ Visualization
- **RViz** displays:
  - Robot pose
  - LiDAR scans
  - Navigation path
- Camera feed shows real-time crack detection output

---

## 📊 Model Details

- **Model Type**: Convolutional Neural Network (CNN)
- **Input**: RGB wall surface images
- **Output**: Crack / No-Crack classification with confidence score
- **Inference Mode**: Real-time, frame-by-frame prediction

---

## ⚠️ Limitations

- Crack detection performance may degrade under poor lighting, occlusion, or extremely fine cracks  
- Model trained on a limited dataset  
- Crack localization is 2D only (no depth estimation)

---

## 🔮 Future Improvements

- Expand training dataset for improved robustness  
- Integrate depth sensing for crack severity estimation  
- Optimize inference using ONNX / TensorRT  
- Add cloud-based monitoring dashboard  
- Multi-robot inspection coordination  

