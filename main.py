import cv2
import tensorflow as tf
import numpy as np
from fastapi import FastAPI

from utils import convert, prepare

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Buslight recognition API"}


@app.get("/status")
async def get_status():
    status, confidence = getPrediction()
    error = None if status is not None else "Camera not detected"
    return {"status": status, "confidence": confidence, "error": error}

model = tf.keras.models.load_model("mlp.model")

def getPrediction():
    for i in range(10):
        cap = cv2.VideoCapture(2)

        _, frame = cap.read()
        # this is required for macbook webcams
        if (not isinstance(frame, np.ndarray)):
            continue

        frame = convert(frame)
        prediction = model.predict(prepare(frame))

        cap.release()
        break

    if frame is None:
        return None, None

    # item() is for converting the numpy float to python float
    if prediction[0][0] < prediction[0][1]:
        result = True
        confidence = prediction[0][0].item()
    else:
        result = False
        confidence = prediction[0][1].item()
    return result, round(confidence, 2)
