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
    return {"status": getPrediction()}


model = tf.keras.models.load_model("mlp.model")


def getPrediction():
    while True:
        cap = cv2.VideoCapture(2)

        _, frame = cap.read()
        print(frame)
        # this is required for macbook webcams
        if (not isinstance(frame, np.ndarray)):
            continue
        frame = convert(frame)

        prediction = model.predict(prepare(frame))
        print(prediction)

        cap.release()
        break
    result = True if prediction[0][0] < prediction[0][1] else False
    return result

    if prediction[0][0] < prediction[0][1]:
        result = True
        confidence = prediction[0][0]
    else
        result = False
        confidence = prediction[0][1]
    return result, confidence
