import pandas as pd


def predict_default(model, data: dict) -> dict:
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"churn_probability": None, "threshold": 0.5, "prediction": prediction}


def predict_with_threshold(model, data: dict, threshold: float) -> dict:
    df = pd.DataFrame([data])
    probs = model.predict_proba(df)[:, 1]
    prediction = (probs > threshold).astype(int)
    return {
        "churn_probability": float(probs[0]),
        "threshold": threshold,
        "churn_prediction": int(prediction[0]),
    }
