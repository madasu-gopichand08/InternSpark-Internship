# Driver Drowsiness Detection API

A containerized REST API for driver drowsiness detection using a deep learning model based on **MobileNetV2, LSTM, and Temporal Attention**.

This project was developed as part of **Task 3 of the InternSpark Artificial Intelligence Internship**.

**Author:** Gopichand Madasu

---

## Overview

The API deploys the trained Driver Drowsiness Detection model developed in Task 2.

The model accepts a sequence of **5 facial image frames**, preprocesses them, and returns a drowsiness prediction with its probability.

### Model Pipeline

```text
5 Facial Frames
      ↓
Grayscale
      ↓
CLAHE
      ↓
Mean Brightness Normalization
      ↓
96 × 96
      ↓
MobileNetV2
      ↓
LSTM
      ↓
Temporal Attention
      ↓
Drowsy / Not Drowsy
