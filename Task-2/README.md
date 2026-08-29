# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using MobileNetV2(CNN pretrained model),
LSTM, temporal attention, facial landmarks, and behavioral cues.

## Overview

This project detects driver drowsiness from facial video sequences.
A pretrained MobileNetV2 network is used for spatial feature
extraction, followed by an LSTM for temporal modeling and a custom
attention mechanism for emphasizing important temporal features.

For real-time webcam inference, the system additionally uses facial
landmarks to calculate behavioral cues such as Eye Aspect Ratio (EAR)
and Mouth Aspect Ratio (MAR), combined with temporal decision logic
and an audible warning system.

## Architecture

Webcam
↓
Face Detection
↓
CLAHE + Mean Normalization
↓
5-frame sequence
↓
MobileNetV2
↓
LSTM
↓
Temporal Attention
↓
Drowsiness Classification

Additional webcam branch:
Face Landmarks → EAR / MAR → Temporal Rules

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- MobileNetV2
- LSTM
- Temporal Attention

## Dataset

Driver Drowsiness Dataset (DDD)

The dataset is not included in this repository.

## Model Configuration

Input:
- 5 frames
- 96 × 96
- grayscale

Backbone:
- MobileNetV2 with ImageNet weights

Temporal model:
- LSTM (128 units)

Attention:
- Custom temporal attention layer

Output:
- Drowsy
- Not Drowsy

## Results

The deep-learning model was evaluated using a subject-independent
test set of 2,491 temporal sequences.

| Metric | Score |
|---|---:|
| Accuracy | 71% |
| Macro F1-score | 71% |
| Weighted F1-score | 71% |

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Not Drowsy | 55% | 98% | 70% |
| Drowsy | 98% | 57% | 72% |

## Real-Time Webcam System

For real-time deployment, the system extends the trained deep-learning
model with facial behavioral analysis.

The webcam pipeline combines:

- MobileNetV2 for spatial feature extraction
- LSTM for temporal behavior modelling
- Temporal Attention for important-frame weighting
- Eye Aspect Ratio (EAR) for sustained eye closure
- Mouth Aspect Ratio (MAR) for yawning detection
- Temporal persistence and recovery logic
- Audible warning using a generated alarm

This hybrid approach is designed to provide more responsive real-time
drowsiness detection than relying on the deep-learning classifier alone.

During manual real-time testing, the system successfully responded to
awake, eye-closure, yawning, and recovery scenarios under the tested
webcam conditions.


## Limitations

The current prototype is designed for real-time detection using a
single visible face in a webcam environment. Performance may vary
with extreme camera conditions, unusual face orientations, or
scenarios outside the training data.

## Future Improvements

- Larger webcam-specific dataset
- Better domain adaptation
- More robust face/eye preprocessing
- Additional temporal behavioral features

## Author

Madasu Gopi Chand
