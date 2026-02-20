import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI
import uvicorn
import joblib
from predict import predict_with_threshold

app = FastAPI()

model = joblib.load("models/model.pkl")


@app.get("/predict")
def predict(data: dict) -> dict:
    prediction_output = predict_with_threshold(model=model, data=data, threshold=0.4)
    return prediction_output


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
