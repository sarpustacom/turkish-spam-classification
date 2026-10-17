"""Minimal Flask service for the Turkish SMS spam classifier."""

import json
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
MAX_MESSAGE_LENGTH = 5000


def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(f"Model files not found in {MODEL_DIR}/. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    classes = list(model.classes_)
    if "spam" not in classes:
        raise RuntimeError(f"Loaded model has no 'spam' class: {classes}")

    return model, float(metadata["threshold"]), classes.index("spam")


# Loaded once at startup; the API never retrains the model.
MODEL, THRESHOLD, SPAM_INDEX = load_model()

app = Flask(__name__)


def error(message: str, status: int):
    return jsonify({"error": message}), status


@app.errorhandler(HTTPException)
def handle_http_error(exc: HTTPException):
    return error(exc.description, exc.code)


@app.get("/")
def index():
    return render_template("index.html", max_length=MAX_MESSAGE_LENGTH)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.post("/api/v1/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Request body must be a JSON object", 400)

    message = data.get("message")
    if not isinstance(message, str):
        return error("'message' must be a string", 400)

    message = message.strip()
    if not message:
        return error("'message' must not be empty", 400)
    if len(message) > MAX_MESSAGE_LENGTH:
        return error(f"'message' must be at most {MAX_MESSAGE_LENGTH} characters", 400)

    try:
        spam_score = float(MODEL.predict_proba([message])[0, SPAM_INDEX])
    except Exception:
        # Deliberately not logging the message content.
        return error("Prediction failed", 500)

    label = "spam" if spam_score >= THRESHOLD else "ham"
    return jsonify({"label": label, "spam_score": round(spam_score, 4)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
