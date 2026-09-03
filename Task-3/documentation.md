# Driver Drowsiness Detection API

**Author:** Gopichand Madasu  
**Project:** InternSpark — Task 3

## 1. Project Overview

The Driver Drowsiness Detection API is a REST API that deploys the trained Driver Drowsiness Detection model developed in Task 2.

The system accepts a sequence of five facial image frames, applies the required preprocessing, passes the sequence through the trained MobileNetV2 + LSTM + Temporal Attention model, and returns a drowsiness prediction with its probability.

Task 3 focuses on converting the trained model into a locally accessible API and containerizing the application using Docker.

## 2. Objectives

- Deploy the trained driver drowsiness model through a REST API.
- Accept five facial frames as model input.
- Apply the same preprocessing required by the trained model.
- Return a Drowsy or Not Drowsy prediction.
- Return the model's prediction probability.
- Provide an interactive API interface through FastAPI Swagger documentation.
- Containerize the complete application using Docker.
- Verify the API both locally and inside the Docker container.

## 3. System Architecture

```text
Five Image Frames
        |
        v
FastAPI /predict
        |
        v
Image Preprocessing
        |
        +--> Grayscale
        |
        +--> CLAHE
        |
        +--> Mean Brightness Normalization
        |
        +--> Resize to 96 × 96
        |
        +--> Normalize to 0–1
        |
        v
Input Shape
(1, 5, 96, 96, 1)
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
Sigmoid Classification
        |
        v
Drowsy / Not Drowsy
        |
        v
JSON Response
```

The complete application is packaged inside a Docker container:
```text
Docker
  |
  +-- FastAPI
  |
  +-- Model Architecture
  |
  +-- Trained Model Weights
  |
  +-- Python Dependencies
```
## 4. Model & Input
The deployed model is the trained model from Task 2.

Model Architecture
```text
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
```
MobileNetV2 is used for visual feature extraction from each frame.

The LSTM models temporal relationships across the five frames.

The custom Temporal Attention layer produces a weighted representation of the temporal features before classification.

Model Input

The API requires exactly five image frames.

Each frame is converted to:
96 × 96 × 1
The complete model input is:
(1, 5, 96, 96, 1)
Model Output

The model produces one sigmoid probability.

The API converts the probability into:
0 → Not Drowsy
1 → Drowsy
The deployment threshold currently used by the API is:
0.5

## 5. Image Preprocessing
The API applies the preprocessing required by the deployed model.
```text
Input Image
     ↓
Grayscale Conversion
     ↓
CLAHE
     ↓
Mean Brightness Normalization
     ↓
96 × 96 Resize
     ↓
0–1 Normalization
```
Grayscale

Each uploaded image is converted to grayscale.

CLAHE

Contrast Limited Adaptive Histogram Equalization is applied to improve local contrast.

Mean Brightness Normalization

The processed image is shifted toward a target mean brightness of: 128
while keeping pixel values within the valid 0–255 range.

Resize

Each frame is resized to:

96 × 96
Normalization

Pixel values are converted from the 0–255 range to:

0.0 – 1.0

by dividing by 255.

## 6. FastAPI Implementation

The API is implemented using FastAPI.

The main application file is:

app.py

The application:

Loads the model architecture.
Loads the trained model weights.
Provides API endpoints.
Receives uploaded image frames.
Preprocesses the frames.
Performs model inference.
Returns the prediction as JSON.

The trained weights are stored in:

model/
└── drowsiness_model_final_v2.weights.h5

The model architecture is recreated in app.py so that the verified .weights.h5 artifact can be loaded reliably.

## 7. API Endpoints

### GET /

Checks whether the API is running.

Example response:

{
  "message": "Driver Drowsiness Detection API",
  "status": "running",
  "model": "MobileNetV2 + LSTM + Temporal Attention",
  "sequence_length": 5,
  "image_size": 96
}

### GET /model-info

Returns information about the deployed model.

Example response:

{
  "model_name": "DriverDrowsiness_API_Model",
  "input_shape": [null, 5, 96, 96, 1],
  "output_shape": [null, 1],
  "image_size": 96,
  "sequence_length": 5,
  "classes": [
    "Not Drowsy",
    "Drowsy"
  ],
  "threshold": 0.5
}

### POST /predict

Accepts five image files:

frame1
frame2
frame3
frame4
frame5

The five frames are processed as one temporal sequence.

Example response for a drowsy sequence:

{
  "prediction": "Drowsy",
  "probability": 0.999999,
  "threshold": 0.5,
  "frames_processed": 5
}

Example response for a non-drowsy sequence:

{
  "prediction": "Not Drowsy",
  "probability": 0.000001,
  "threshold": 0.5,
  "frames_processed": 5
}
## 8. Prediction Workflow

The API prediction workflow is:
```text
Client
  |
  v
POST /predict
  |
  v
Upload 5 Frames
  |
  v
Validate Input
  |
  v
Decode Images
  |
  v
Preprocess Each Frame
  |
  v
Create Sequence
(1, 5, 96, 96, 1)
  |
  v
Run Model
  |
  v
Obtain Probability
  |
  v
Apply Threshold
  |
  v
Return JSON Response
```

The API requires exactly five frames because the deployed model was trained with a sequence length of five.

## 9. Docker Containerization

The application is containerized using Docker.

The Dockerfile defines:

Base Python environment
Required system libraries
Python dependencies
Application files
Model files
API port
API startup command

The Docker container runs FastAPI using Uvicorn on port 8000.

The project is built using:

docker build -t driver-drowsiness-api .

The container is started using:

docker run -p 8000:8000 --name driver-drowsiness-container driver-drowsiness-api

The API is then accessible through:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

## 10. Installation & Usage
Local Python Execution

Install the required packages:

pip install -r requirements.txt

Start the API:

python app.py

The API runs at:

http://127.0.0.1:8000
Docker Execution

Build the image:

docker build -t driver-drowsiness-api .

Run the container:

docker run -p 8000:8000 --name driver-drowsiness-container driver-drowsiness-api

Open the Swagger interface:

http://127.0.0.1:8000/docs

Use POST /predict to upload five image frames.

## 11. Testing & Results

The API was tested locally before containerization.

Both drowsy and non-drowsy sequences were successfully classified through the /predict endpoint.

The API returned HTTP status code:

200 OK

for successful prediction requests.

Local API Tests
Drowsy Test
Prediction    : Drowsy
Probability   : 0.999999
Frames        : 5
HTTP Status   : 200
Not Drowsy Test
Prediction    : Not Drowsy
Probability   : 0.000001
Frames        : 5
HTTP Status   : 200
Docker API Tests

The same inference workflow was tested after deploying the API inside Docker.

Drowsy Test
Prediction    : Drowsy
Probability   : 0.999999
Frames        : 5
HTTP Status   : 200
Not Drowsy Test
Prediction    : Not Drowsy
Probability   : 0.000001
Frames        : 5
HTTP Status   : 200

These tests verified that:

The trained model can be loaded inside the container.
The API accepts five image files.
Image preprocessing works inside the container.
Model inference works inside the container.
The API returns valid JSON responses.


## 12. Limitations & Future Improvements
Limitations

The current implementation is a local prototype for model deployment and is not intended to represent a production-grade cloud service or a safety-certified driver monitoring system.

The API expects exactly five image frames and currently processes one sequence per prediction request.

The deployment uses CPU-based inference inside the Docker container.

Future Improvements

Future versions could include:

A production deployment on a cloud platform.
Batch and asynchronous prediction support.
Authentication and API security.
Health and monitoring endpoints.
Improved error handling and request validation.
API support for video streams or real-time frame sequences.
Model optimization for lower inference latency.
Automated testing and continuous integration.
