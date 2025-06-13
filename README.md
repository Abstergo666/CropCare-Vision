# 🍃 Real-Time Leaf Disease Detection using Deep Learning | B.Tech Final Year Project

This repository contains the code for real-time spinach leaf disease detection using a custom-trained Convolutional Neural Network (CNN) model. The system captures live video from an IP camera, performs classification using a Keras model, and optionally streams the annotated video over RTSP.

---

## 🎯 Objective

To design and implement an AI-based smart monitoring system that detects common diseases in spinach leaves in real-time using deep learning, making it suitable for applications like vertical hydroponics and precision agriculture.

---

## 🧠 Model Overview

- Framework: TensorFlow / Keras
- Input: 224x224 RGB images
- Output Classes:
  - **Anthracnose**
  - **Bacterial Spot**
  - **Downy Mildew**
  - **Healthy Leaf**
  - **Pest Damage**

Model File: `spinach_leaf_model.h5`

---

## 🛠️ Tech Stack

| Component        | Description                             |
|------------------|-----------------------------------------|
| 📷 IP Camera     | Live input stream (Android IP Webcam app) |
| 🧠 TensorFlow    | CNN model for multi-class leaf classification |
| 🔁 OpenCV        | Video processing and frame annotation   |
| 📡 RTSP Output   | Optional RTSP stream using FFmpeg or GStreamer |

---


