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
```

## Technologies:
Python
TensorFlow / Keras
MobileNetV2
LSTM
Temporal Attention
OpenCV
FastAPI
Uvicorn
Docker

## Project Structure
TASK-3/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── documentation.md
│
└── model/
    └── drowsiness_model_final_v2.weights.h5

## Run Without Docker

Start the API using:

python app.py

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

## Run With Docker

Build the Docker image:

docker build -t driver-drowsiness-api .

Run the container:

docker run -p 8000:8000 --name driver-drowsiness-container driver-drowsiness-api

The API will then be available at:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs


## API Endpoints
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

POST /predict

Accepts five image files:

frame1
frame2
frame3
frame4
frame5

and returns the prediction.

Example response:

{
  "prediction": "Drowsy",
  "probability": 0.999999,
  "threshold": 0.5,
  "frames_processed": 5
}

For an alert sequence, the API returns:

{
  "prediction": "Not Drowsy",
  "probability": 0.000001,
  "threshold": 0.5,
  "frames_processed": 5
}

## Sample Request

The `/predict` endpoint accepts exactly five image frames.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "frame1=@drowsy1.png" \
  -F "frame2=@drowsy2.png" \
  -F "frame3=@drowsy3.png" \
  -F "frame4=@drowsy4.png" \
  -F "frame5=@drowsy5.png"
```
Example response:

{
  "prediction": "Drowsy",
  "probability": 0.999999,
  "threshold": 0.5,
  "frames_processed": 5
}


## Docker Verification

The containerized API was tested locally using both classes.

Drowsy Test
HTTP 200
Prediction: Drowsy
Probability: 0.999999
Frames processed: 5
Not Drowsy Test
HTTP 200
Prediction: Not Drowsy
Probability: 0.000001
Frames processed: 5

These tests confirm that the model can be loaded and used successfully through the FastAPI service running inside Docker.

## Author

Gopichand Madasu

InternSpark — Artificial Intelligence Internship
