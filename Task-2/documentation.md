# Driver Drowsiness Detection System — Technical Documentation

## 1. Project Overview

The Driver Drowsiness Detection System is a real-time computer vision and deep learning application designed to identify signs of driver drowsiness from facial video.

The system consists of two major components:

1. A deep learning model trained on the Driver Drowsiness Dataset (DDD).
2. A real-time webcam inference system that combines the trained model with facial behavioral analysis.

The deep learning pipeline uses a pretrained MobileNetV2 backbone for visual feature extraction, an LSTM for temporal modeling, and a custom attention layer for temporal feature weighting.

The webcam application additionally uses OpenCV and MediaPipe facial landmarks to estimate eye and mouth behavior through Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).

---

## 2. System Architecture

### Training Pipeline

```text
Driver Drowsiness Dataset (DDD)
              ↓
       Filename Parsing
              ↓
        Run Detection
              ↓
      Temporal Windows
              ↓
     Subject-Level Split
              ↓
   Image Normalization
   ├── Grayscale
   ├── CLAHE
   └── Mean Brightness Normalization
              ↓
        5-Frame Sequence
              ↓
      MobileNetV2 CNN
              ↓
          Dense(128)
              ↓
          LSTM(128)
              ↓
       Temporal Attention
              ↓
          Dense(64)
              ↓
        Sigmoid Output
              ↓
     Drowsy / Not Drowsy
text''' 
3. Dataset

The project uses the Driver Drowsiness Dataset (DDD).

The dataset is organized into two primary classes:

Drowsy
Non Drowsy

The training pipeline initially parses filenames to extract subject information, frame sequence numbers, labels, and file paths.

The dataset processing identified:

41,793 image files
28 subjects
1,172 continuous sub-runs

Continuous runs were detected using sequence-number gaps so that unrelated image sequences were not combined into the same temporal window.

4. Dataset Parsing

Each image is converted into a metadata record containing:
subject
sequence number
label
filepath

The label mapping used by the training pipeline is:
Drowsy     → 1
Non Drowsy  → 0
A run_key is created from the subject, label, and detected run identifier.
This allows temporal windows to be constructed only from frames belonging to the same recording run.

