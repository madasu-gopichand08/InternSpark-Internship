# Driver Drowsiness Detection System

## Technical Documentation

**Author:** Gopichand Madasu  
**Project:** InternSpark — Task 2  
**Domain:** Deep Learning / Computer Vision

---

# 1. Project Overview

The Driver Drowsiness Detection System is a computer vision and deep learning based system designed to identify signs of driver drowsiness from facial video sequences.

The project consists of two major components:

1. A deep learning training pipeline developed using the Driver Drowsiness Dataset (DDD).
2. A real-time webcam detection system developed locally using the trained model together with facial behavioral analysis.

The deep learning model uses a pretrained MobileNetV2 backbone for visual feature extraction, an LSTM for temporal modeling, and a custom attention mechanism for emphasizing important temporal features.

The real-time webcam application additionally uses OpenCV and MediaPipe facial landmarks to analyze eye closure and yawning behavior using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

---

# 2. Objectives

The main objectives of the project are:

- Detect driver drowsiness from facial video sequences.
- Extract meaningful visual features using a pretrained CNN.
- Model temporal changes across consecutive facial frames.
- Use temporal attention to emphasize important parts of a sequence.
- Evaluate the model using subjects that were not used during training.
- Deploy the trained model for real-time webcam inference.
- Incorporate facial behavioral signals such as eye closure and yawning.
- Provide an audible warning when drowsiness is detected.

---

# 3. System Architecture

The complete system consists of a training pipeline and a real-time inference pipeline.

## 3.1 Training Pipeline

```text
Driver Drowsiness Dataset (DDD)
             |
             v
      Filename Parsing
             |
             v
        Run Detection
             |
             v
      Temporal Windows
             |
             v
     Subject-Level Split
             |
             v
      Image Preprocessing
             |
             +----------------------+
             |                      |
             v                      |
         Grayscale                 |
             |                      |
             v                      |
            CLAHE                  |
             |                      |
             v                      |
    Mean Brightness               |
      Normalization               |
             |                      |
             +----------+-----------+
                        |
                        v
                 5-Frame Sequence
                        |
                        v
                  MobileNetV2
                        |
                        v
                   Dense(128)
                        |
                        v
                    LSTM(128)
                        |
                        v
               Temporal Attention
                        |
                        v
                    Dense(64)
                        |
                        v
                     Sigmoid
                        |
                        v
              Drowsy / Not Drowsy

3.2 Real-Time Webcam Pipeline
Webcam
  |
  v
DNN Face Detection
  |
  v
Face Preprocessing
  |
  v
5-Frame Buffer
  |
  v
MobileNetV2
  |
  v
LSTM
  |
  v
Temporal Attention
  |
  v
**CNN Drowsiness Signal**

