"""Minimal Flask service for the Turkish SMS spam classifier (V3 BiLSTM model)."""

import csv
import json
import math
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl  # file locking across processes (macOS/Linux)
except ImportError:  # Windows: only the in-process lock is used
    fcntl = None

import keras
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

# Registers the custom Turkish standardize function the saved model needs to load.
import train_bilstm  # noqa: F401

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "spam_classifier_v3.keras"
METADATA_PATH = MODEL_DIR / "metadata_v3.json"
MAX_MESSAGE_LENGTH = 5000

FEEDBACK_PATH = Path(__file__).parent / "data" / "user_feedback.csv"
FEEDBACK_LOCK_PATH = FEEDBACK_PATH.with_name(FEEDBACK_PATH.name + ".lock")
FEEDBACK_FIELDS = ["message", "label", "updated_at"]
FEEDBACK_LABELS = ("ham", "spam")
# Cells starting with these can run as formulas in Excel/Sheets. "'" is included so
# the escaping is reversible: a stored value starting with "'" always had one added.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "'")
_feedback_lock = threading.Lock()


def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            f"V3 model files not found in {MODEL_DIR}/. Run `python train_bilstm.py` first."
        )

    model = keras.models.load_model(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())

    # Expect raw strings in, one spam probability out.
    (input_tensor,) = model.inputs
    (output_tensor,) = model.outputs
    if input_tensor.dtype != "string" or tuple(output_tensor.shape) != (None, 1):
        raise RuntimeError(
            f"Unexpected V3 model signature: input {input_tensor.dtype}, output {output_tensor.shape}"
        )
    if metadata.get("positive_label") != "spam":
        raise RuntimeError("metadata_v3.json: model output is not the spam probability")

    threshold = float(metadata["threshold"])
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"metadata_v3.json: invalid threshold {threshold}")

    return model, threshold


# Loaded once at startup; the API never retrains the model.
MODEL, THRESHOLD = load_model()

app = Flask(__name__)


def error(message: str, status: int):
    return jsonify({"error": message}), status


def read_message(data):
    """Validate the 'message' field. Returns (message, None) or (None, error response)."""
    if not isinstance(data, dict):
        return None, error("Request body must be a JSON object", 400)

    message = data.get("message")
    if not isinstance(message, str):
        return None, error("'message' must be a string", 400)

    message = message.strip()
    if not message:
        return None, error("'message' must not be empty", 400)
    if len(message) > MAX_MESSAGE_LENGTH:
        return None, error(f"'message' must be at most {MAX_MESSAGE_LENGTH} characters", 400)

    return message, None


def escape_csv_cell(value: str) -> str:
    return "'" + value if value.startswith(FORMULA_PREFIXES) else value


def unescape_csv_cell(value: str) -> str:
    return value[1:] if value.startswith("'") else value


@contextmanager
def feedback_file_lock():
    """Serialize writes between threads and between processes (e.g. several workers)."""
    with _feedback_lock:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_LOCK_PATH, "w") as lock_file:
            if fcntl:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)


def save_feedback(message: str, label: str) -> bool:
    """Insert or update the label for a message. Returns True if it already existed."""
    with feedback_file_lock():
        rows = {}
        if FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows[unescape_csv_cell(row["message"])] = row

        updated = message in rows
        rows[message] = {"label": label, "updated_at": datetime.now(timezone.utc).isoformat()}

        # Write to a temp file and swap it in, so a failed write never corrupts the CSV.
        tmp_path = FEEDBACK_PATH.with_name(FEEDBACK_PATH.name + ".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
            writer.writeheader()
            for text, row in rows.items():
                writer.writerow(
                    {"message": escape_csv_cell(text), "label": row["label"], "updated_at": row["updated_at"]}
                )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, FEEDBACK_PATH)

    return updated


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
    message, error_response = read_message(request.get_json(silent=True))
    if error_response:
        return error_response

    try:
        # Preprocessing is inside the model; it takes a (batch, 1) string tensor.
        output = MODEL(tf.constant([[message]]), training=False)
        spam_score = float(output.numpy()[0, 0])
        if not math.isfinite(spam_score):
            raise ValueError("non-finite model output")
    except Exception:
        # Deliberately not logging the message content.
        return error("Prediction failed", 500)

    label = "spam" if spam_score >= THRESHOLD else "ham"
    return jsonify({"label": label, "spam_score": round(spam_score, 4)})


@app.post("/feedback")
def feedback():
    data = request.get_json(silent=True)
    message, error_response = read_message(data)
    if error_response:
        return error_response

    label = data.get("label")
    if label not in FEEDBACK_LABELS:
        return error("'label' must be 'ham' or 'spam'", 400)

    try:
        updated = save_feedback(message, label)
    except Exception as exc:
        # Log only the error type, never the message content.
        app.logger.error("Saving feedback failed: %s", type(exc).__name__)
        return error("Could not save feedback", 500)

    return jsonify({"status": "saved", "label": label, "updated": updated})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
