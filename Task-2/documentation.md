# Driver Drowsiness Detection System

**Author:** Madasu Gopi Chand

## 1. Project Overview

The Driver Drowsiness Detection System is a real-time computer vision and deep learning system designed to detect driver drowsiness from facial video. The project combines a MobileNetV2-based temporal deep learning model with LSTM and temporal attention, along with facial behavioral analysis using EAR and MAR for real-time webcam detection.

## 2. Objectives

- Detect drowsy and non-drowsy states from facial video sequences.
- Extract visual features using pretrained MobileNetV2.
- Model temporal facial changes using LSTM.
- Use temporal attention to emphasize important frames.
- Evaluate the model using subject-independent testing.
- Deploy the system for real-time webcam detection.
- Provide an audible warning when drowsiness is detected.

## 3. System Architecture

```text
DDD Dataset
    ↓
Data Preparation
    ↓
Run Detection
    ↓
5-Frame Temporal Windows
    ↓
Image Preprocessing
    ↓
MobileNetV2
    ↓
Dense(128)
    ↓
LSTM(128)
    ↓
Temporal Attention
    ↓
Dense(64)
    ↓
Sigmoid
    ↓
Drowsy / Not Drowsy
```

For real-time detection, the webcam pipeline additionally uses DNN face detection and MediaPipe facial landmarks with EAR/MAR-based temporal analysis.

## 4. Dataset & Data Preparation

The project uses the **Driver Drowsiness Dataset (DDD)**, containing `Drowsy` and `Non Drowsy` classes.

The dataset contains **41,793 images from 28 subjects**. Image filenames are parsed to extract subject identity, frame sequence number, label, and file path.

Continuous frame runs are identified using sequence-number gaps so that unrelated recording segments are not combined. The processed metadata is stored in `ddd2_runs_manifest.csv`.

The dataset is divided at the subject level into training, validation, and test subjects to evaluate generalization to unseen individuals.

## 5. Temporal Sequence Generation

The model processes multiple frames instead of individual images.

The final configuration uses:

- Sequence length: **5 frames**
- Window stride: **2 frames**
- Image size: **96 × 96**

Temporal windows are generated within each recording run so that frames from unrelated recordings are never combined.

## 6. Image Preprocessing

Each image undergoes the following preprocessing:

```text
Original Image
    ↓
Grayscale Conversion
    ↓
CLAHE
    ↓
Mean Brightness Normalization
    ↓
Resize to 96 × 96
    ↓
Normalize to 0–1
```

CLAHE is used to improve local contrast, while mean-brightness normalization targets a mean intensity of 128 — this step was added specifically to remove a systematic brightness difference found between the two classes during dataset auditing, which could otherwise be learned as a shortcut instead of genuine facial cues.

Training data also uses mild augmentation through horizontal flipping and limited brightness scaling.

## 7. Model Architecture

The final deep learning model consists of:

```text
5 × 96 × 96 × 1 Input
        ↓
Grayscale → 3 Channels
        ↓
MobileNetV2 (pretrained, frozen)
        ↓
Dense(128)
        ↓
LSTM(128)
        ↓
Temporal Attention
        ↓
Dense(64)
        ↓
Dropout
        ↓
Sigmoid
```

MobileNetV2 is used as the pretrained spatial feature extractor. LSTM models temporal relationships between frames, while the custom attention layer assigns importance to different temporal steps.

The output is a binary prediction:

- `0` → Not Drowsy
- `1` → Drowsy

## 8. Model Training & Evaluation

The model is trained using the **Adam optimizer** with a learning rate of `0.0001` and **binary cross-entropy** loss.

Class weighting, model checkpointing, early stopping, and learning-rate reduction are used during training.

The final subject-independent test set contains **2,491 temporal sequences**.

### Test Results

| Metric | Score |
|---|---|
| Accuracy | 71% |
| Macro F1-score | 71% |
| Weighted F1-score | 71% |

### Class-wise Results

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Not Drowsy | 55% | 98% | 70% |
| Drowsy | 98% | 57% | 72% |

**Confusion matrix:**

```text
[[842,  19],
 [700, 930]]
```

### A Note on Reproducibility

Across multiple training runs with an identical configuration (same architecture, same seed setting, same data), test accuracy varied between approximately 56% and 72%, with precision/recall balance also shifting between runs (e.g., which class achieved higher recall). This variance is attributed to two factors: (1) the model reaches very high training accuracy (>99%) within the first 1–2 epochs, meaning the specific epoch selected as "best" by early stopping is sensitive to small early differences between runs, and (2) certain GPU-accelerated operations (particularly in cuDNN's LSTM and convolution kernels) are not fully deterministic even with fixed random seeds. The results reported above correspond to a single specific, fully saved and version-controlled training run, whose model weights, training curves, and classification report were all generated together in one execution to ensure internal consistency.

## 9. Real-Time Webcam System

The trained model is integrated into a local Jupyter Notebook for real-time webcam detection.

The webcam system uses OpenCV and a DNN-based face detector to locate the face. The detected face is processed using the same general preprocessing approach used during training and stored in a 5-frame temporal buffer.

The sequence is then passed through the trained MobileNetV2 + LSTM + Attention model for drowsiness prediction.

## 10. EAR / MAR Behavioral Analysis

The webcam system additionally uses **MediaPipe FaceLandmarker** to obtain facial landmarks.

Two behavioral measurements are calculated:

- **EAR (Eye Aspect Ratio):** used to detect sustained eye closure.
- **MAR (Mouth Aspect Ratio):** used to detect sustained mouth opening associated with yawning.

Temporal conditions are applied to these measurements to avoid triggering drowsiness from a single frame (e.g., a normal blink).

The CNN-based prediction and facial behavioral signals are combined to determine the final real-time drowsiness state — the system also includes a fast-recovery mechanism that resets the CNN's temporal buffer when the behavioral signals indicate a clearly alert state, preventing stale sliding-window predictions from causing delayed recovery after a drowsy episode ends.

## 11. Installation & Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Place the trained model in the project's `model/` directory and open the webcam inference notebook.

Run the webcam detection function and allow the camera to capture the driver's face.

The system displays:

- `ALERT`

or

- `DROWSY`

and activates:

- `!! WAKE UP !!`

with an audible alarm when sustained drowsiness is detected.

Press **Q** to stop the webcam application.

## 12. Results, Limitations & Future Improvements

The deep learning model achieved 71% accuracy on the subject-independent DDD test set, with 98% precision for the Drowsy class.

The complete real-time prototype combines deep learning with EAR/MAR-based behavioral analysis and temporal decision logic. During manual webcam testing, the system successfully responded to tested awake and drowsy scenarios, including sustained eye closure and yawning.

**Known limitations:**

- The current system is a prototype and has not been evaluated as a safety-certified driver monitoring system.
- Performance may vary in conditions substantially different from those represented in the available training and testing data (e.g., extreme lighting, unusual head angles, or camera hardware differences).
- As noted in Section 8, the training process shows run-to-run variance in final accuracy; the reported metrics correspond to one specific, fully documented training run rather than a guaranteed fixed outcome of the training procedure.
- One test subject showed disproportionately low accuracy during evaluation, which on inspection appeared to correlate with session-level (rather than frame-level) labeling in the source dataset — some frames labeled "drowsy" showed a visually alert expression.

**Future improvements:**

- Increasing real-world training data and subject diversity.
- Improving domain adaptation across different cameras/lighting setups.
- Incorporating additional head-pose information.
- Performing larger-scale, real-world evaluation beyond manual webcam testing.
- Investigating training-run variance further, e.g., through deterministic GPU operation settings or averaging predictions across multiple independently trained models (ensembling).

## 13. Author

Gopichand Madasu
