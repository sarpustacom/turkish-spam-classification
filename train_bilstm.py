"""Train the V3 Turkish SMS spam classifier (TextVectorization + BiLSTM, TensorFlow/Keras).

Data: data/turkish_spam_combined.csv (columns: message, label), plus user corrections from
data/user_feedback.csv if present. The split is defined on the main dataset only, so the
validation/test sets stay fixed; feedback only ever adds to or relabels training data.
Both CSV files are only read, never written.
For a fair comparison, a TF-IDF + Logistic Regression model (the V2 architecture from
train.py) is trained on exactly the same split, and the production V2 model is also
scored on the same test set. Does not touch the production model used by app.py.
"""

import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from train import (
    LABELS,
    MODEL_DIR,
    RANDOM_STATE,
    build_pipeline,
    evaluate,
    print_metrics,
    select_threshold,
    spam_probabilities,
)

DATA_PATH = Path(__file__).parent / "data" / "turkish_spam_combined.csv"
FEEDBACK_PATH = DATA_PATH.with_name("user_feedback.csv")
MODEL_PATH = MODEL_DIR / "spam_classifier_v3.keras"
METADATA_PATH = MODEL_DIR / "metadata_v3.json"
METRICS_PATH = MODEL_DIR / "metrics_v3.json"
SPLIT_PATH = MODEL_DIR / "split_indices.json"

# Production model, scored on the same test set for reference only.
V2_MODEL_PATH = MODEL_DIR / "spam_classifier_v2.joblib"
V2_METADATA_PATH = MODEL_DIR / "metadata.json"

CONFIG = {
    "max_tokens": 20000,
    "sequence_length": 64,  # longest message is 53 words
    "embedding_dim": 128,
    "lstm_units": 64,
    "dropout": 0.5,
    "batch_size": 32,
    "max_epochs": 30,
    "learning_rate": 1e-3,
    "early_stopping_patience": 3,
}


@keras.saving.register_keras_serializable(package="spam_v3")
def turkish_standardize(text):
    """Turkish-aware lowercasing + punctuation removal.

    Keras' default standardization lowercases ASCII only, so "İ", "Ş", "Ç"
    would stay uppercase. "I" maps to "ı" and "İ" to "i" as in Turkish.
    """
    text = tf.strings.regex_replace(text, "İ", "i")
    text = tf.strings.regex_replace(text, "I", "ı")
    text = tf.strings.lower(text, encoding="utf-8")
    return tf.strings.regex_replace(text, r"[[:punct:]]", " ")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    missing = {"message", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    n_raw = len(df)
    df = df.dropna(subset=["message", "label"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df["message"] != "") & df["label"].isin(LABELS)]
    n_valid = len(df)
    df = df.drop_duplicates(subset="message").reset_index(drop=True)

    print(f"Raw records:        {n_raw}")
    print(f"Invalid/empty:      {n_raw - n_valid}")
    print(f"Duplicates removed: {n_valid - len(df)}")
    print(f"Final records:      {len(df)}")
    return df


def normalize_text(text: str) -> str:
    """Comparison key for duplicates: Unicode NFC, lowercase, single spaces.

    I/İ/ı/i all fold to "i", since SMS often mixes Turkish and ASCII keyboards
    ("KAZANDINIZ" vs "kazandınız"). Used only for comparison, not as model input.
    """
    text = unicodedata.normalize("NFC", text).replace("İ", "i").replace("I", "i")
    return " ".join(text.lower().replace("ı", "i").split())


def load_feedback() -> pd.DataFrame:
    """Read user corrections. Returns an empty frame if the file is missing or empty."""
    empty = pd.DataFrame(columns=["message", "label"])
    if not FEEDBACK_PATH.exists() or FEEDBACK_PATH.stat().st_size == 0:
        print(f"{FEEDBACK_PATH.name}: not found or empty, using the main dataset only")
        return empty

    # keep_default_na=False: messages like "NA" or "null" stay text.
    df = pd.read_csv(FEEDBACK_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = {"message", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{FEEDBACK_PATH.name} is missing expected columns: {missing}")
    if df.empty:
        print(f"{FEEDBACK_PATH.name}: no records, using the main dataset only")
        return empty

    n_raw = len(df)
    # app.py prefixes "'" to cells that could run as spreadsheet formulas; undo it.
    df["message"] = df["message"].map(lambda m: m[1:] if m.startswith("'") else m).str.strip()
    df["label"] = df["label"].str.strip()
    valid = (df["message"] != "") & df["label"].isin(LABELS)
    df = df[valid].copy()

    # If a message was corrected more than once, keep the most recent correction.
    df["key"] = df["message"].map(normalize_text)
    if "updated_at" in df.columns:
        df = df.sort_values("updated_at", kind="stable")
    n_valid = len(df)
    df = df.drop_duplicates(subset="key", keep="last")

    print(f"{FEEDBACK_PATH.name}: {n_raw} records "
          f"({n_raw - n_valid} invalid skipped, {n_valid - len(df)} repeated corrections merged)")
    return df[["message", "label"]].reset_index(drop=True)


def combine_with_feedback(df: pd.DataFrame, split: dict, feedback: pd.DataFrame):
    """Merge feedback into the training data only; validation/test rows stay unchanged.

    Returns one DataFrame (message, label, split, source) and the merge statistics.
    """
    data = df[["message", "label"]].copy()
    data["split"] = ""
    for name in ("train", "validation", "test"):
        data.iloc[split[name], data.columns.get_loc("split")] = name
    data["source"] = "main"
    data["key"] = data["message"].map(normalize_text)

    evaluation = data[data["split"] != "train"]
    eval_keys = set(evaluation["key"])
    train = data[data["split"] == "train"]

    # Training rows that duplicate an evaluation message (after normalization) would leak.
    n = len(train)
    train = train[~train["key"].isin(eval_keys)]
    main_eval_overlap = n - len(train)
    n = len(train)
    train = train.drop_duplicates(subset="key", keep="first")
    main_duplicates = n - len(train)

    fb = feedback.assign(key=feedback["message"].map(normalize_text), split="train", source="feedback")
    in_eval = fb["key"].isin(eval_keys)
    fb_excluded = int(in_eval.sum())
    fb = fb[~in_eval]

    # Feedback wins over the main dataset for the same message.
    overridden = train[train["key"].isin(fb["key"])]
    main_labels = overridden.set_index("key")["label"]
    labels_changed = int((fb.set_index("key")["label"].reindex(main_labels.index) != main_labels).sum())
    train = train[~train["key"].isin(fb["key"])]

    combined = pd.concat([train, fb, evaluation], ignore_index=True).drop(columns="key")
    if not set(combined["label"]) <= set(LABELS):
        raise ValueError(f"Unexpected labels: {set(combined['label']) - set(LABELS)}")

    stats = {
        "main_records": len(df),
        "feedback_records": len(feedback),
        "duplicates_resolved": len(overridden) + main_duplicates,
        "feedback_overrode_main": len(overridden),
        "feedback_labels_changed": labels_changed,
        "feedback_new_messages": len(fb) - len(overridden),
        "feedback_excluded_in_eval_set": fb_excluded,
        "main_train_duplicates_removed": main_duplicates,
        "main_train_rows_overlapping_eval_removed": main_eval_overlap,
        "final_records": len(combined),
    }
    return combined, stats


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash of the cleaned rows, so saved split indices can't be applied to other data."""
    content = "\n".join(df["message"] + "\t" + df["label"])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_or_create_split(df: pd.DataFrame) -> dict:
    """Reuse saved split indices if they match this dataset; otherwise create and save them."""
    fingerprint = dataset_fingerprint(df)
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text())
        if split.get("fingerprint") != fingerprint:
            raise ValueError(
                f"{SPLIT_PATH.name} was made for a different dataset. "
                "Delete it to create a new split."
            )
        print(f"Using saved split indices from {SPLIT_PATH.name}")
        return split

    labels = df["label"].to_numpy()
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=RANDOM_STATE
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=RANDOM_STATE
    )
    split = {
        "dataset": DATA_PATH.name,
        "fingerprint": fingerprint,
        "n_rows": len(df),
        "random_state": RANDOM_STATE,
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    SPLIT_PATH.write_text(json.dumps(split))
    print(f"Saved new split indices to {SPLIT_PATH.name}")
    return split


def build_model(train_texts: np.ndarray) -> keras.Model:
    vectorizer = keras.layers.TextVectorization(
        max_tokens=CONFIG["max_tokens"],
        standardize=turkish_standardize,
        output_mode="int",
        output_sequence_length=CONFIG["sequence_length"],
        name="text_vectorization",
    )
    vectorizer.adapt(train_texts)  # training data only

    inputs = keras.Input(shape=(1,), dtype="string", name="message")
    x = vectorizer(inputs)
    x = keras.layers.Embedding(
        input_dim=vectorizer.vocabulary_size(),
        output_dim=CONFIG["embedding_dim"],
        mask_zero=True,
        name="embedding",
    )(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(CONFIG["lstm_units"]), name="bilstm")(x)
    x = keras.layers.Dropout(CONFIG["dropout"], name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inputs, outputs, name="spam_classifier_v3")
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def make_dataset(texts: np.ndarray, labels: np.ndarray | None = None, shuffle: bool = False,
                 batch_size: int = 256) -> tf.data.Dataset:
    """Batch raw messages as (batch, 1) strings; Keras can't take NumPy string arrays."""
    inputs = tf.constant(texts.astype(str).reshape(-1, 1))
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(inputs)
    else:
        targets = tf.constant((labels == "spam").astype("float32"))
        ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=RANDOM_STATE)
    return ds.batch(batch_size)


def predict_spam(model: keras.Model, texts: np.ndarray) -> np.ndarray:
    return model.predict(make_dataset(texts), verbose=0).ravel()


def train_tfidf_baseline(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """V2 architecture (train.py pipeline) trained on the same split. Not saved."""
    print("\nTraining TF-IDF + Logistic Regression baseline on the same split...")
    baseline = build_pipeline().fit(X_train, y_train)
    threshold = select_threshold(y_val, spam_probabilities(baseline, X_val))
    metrics = evaluate(y_test, spam_probabilities(baseline, X_test), threshold)
    metrics["threshold"] = threshold
    print_metrics(f"TF-IDF + LR baseline, test @ {threshold:.4f}", metrics)
    return metrics


def evaluate_v2(X_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    """Score the production V2 model on the same test set, as app.py uses it."""
    if not (V2_MODEL_PATH.exists() and V2_METADATA_PATH.exists()):
        print("\nV2 model not found; skipping.")
        return None
    v2 = joblib.load(V2_MODEL_PATH)
    threshold = float(json.loads(V2_METADATA_PATH.read_text())["threshold"])
    metrics = evaluate(y_test, spam_probabilities(v2, X_test), threshold)
    metrics["threshold"] = threshold
    metrics["note"] = (
        "V2 was trained outside this script on an unknown split; it may have seen "
        "some of these test messages, so these numbers may be optimistic."
    )
    print_metrics(f"Production V2, test @ {threshold:.4f} (may be optimistic)", metrics)
    return metrics


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    keras.utils.set_random_seed(RANDOM_STATE)

    print(f"Loading {DATA_PATH.name}...")
    df = load_data()
    # The split is defined on the main dataset only, so validation/test stay fixed.
    split = load_or_create_split(df)

    print(f"\nLoading {FEEDBACK_PATH.name}...")
    feedback = load_feedback()
    combined, data_stats = combine_with_feedback(df, split, feedback)

    print("\nCombined dataset:")
    print(f"  Records from {DATA_PATH.name}: {data_stats['main_records']}")
    print(f"  Records from {FEEDBACK_PATH.name}: {data_stats['feedback_records']}")
    print(f"  Duplicates resolved: {data_stats['duplicates_resolved']} "
          f"({data_stats['feedback_overrode_main']} relabeled by feedback, "
          f"{data_stats['feedback_labels_changed']} with a different label; "
          f"{data_stats['main_train_duplicates_removed']} within the main training data)")
    print(f"  New messages from feedback (training only): {data_stats['feedback_new_messages']}")
    print(f"  Feedback excluded (message is in the fixed validation/test set): "
          f"{data_stats['feedback_excluded_in_eval_set']}")
    print(f"  Main training rows removed for overlapping validation/test: "
          f"{data_stats['main_train_rows_overlapping_eval_removed']}")
    print(f"  Final combined size: {data_stats['final_records']}")

    counts = combined["label"].value_counts()
    print("\nClass distribution:")
    for label, count in counts.items():
        print(f"  {label:5s} {count:6d} ({count / len(combined):.2%})")
    if set(counts.index) != set(LABELS):
        raise ValueError("Both 'spam' and 'ham' classes are required for training.")

    parts = {name: combined[combined["split"] == name] for name in ("train", "validation", "test")}
    X_train, X_val, X_test = (parts[n]["message"].to_numpy(dtype=object) for n in parts)
    y_train, y_val, y_test = (parts[n]["label"].to_numpy() for n in parts)
    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Balanced class weights, like class_weight="balanced" in the TF-IDF models.
    weights = compute_class_weight("balanced", classes=np.array(LABELS), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}  # 0 = ham, 1 = spam

    print("\nTraining BiLSTM...")
    model = build_model(X_train)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        make_dataset(X_train, y_train, shuffle=True, batch_size=CONFIG["batch_size"]),
        validation_data=make_dataset(X_val, y_val),
        epochs=CONFIG["max_epochs"],
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=2,
    )
    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    val_proba = predict_spam(model, X_val)
    val_default = evaluate(y_val, val_proba, 0.5)
    threshold = select_threshold(y_val, val_proba)
    val_tuned = evaluate(y_val, val_proba, threshold)
    print(f"\nSelected threshold (max validation spam F1): {threshold:.4f}")
    print_metrics("V3 validation @ 0.5", val_default)
    print_metrics(f"V3 validation @ {threshold:.4f}", val_tuned)

    # Test set is evaluated exactly once, with the threshold chosen on validation.
    test_metrics = evaluate(y_test, predict_spam(model, X_test), threshold)
    print_metrics(f"V3 test @ {threshold:.4f}", test_metrics)

    baseline_test = train_tfidf_baseline(X_train, y_train, X_val, y_val, X_test, y_test)
    v2_test = evaluate_v2(X_test, y_test)

    model.save(MODEL_PATH)

    # Check the saved file reloads and gives the same scores on raw strings.
    reloaded = keras.models.load_model(MODEL_PATH)
    sample = X_test[:64]
    if not np.allclose(predict_spam(reloaded, sample), predict_spam(model, sample), atol=1e-5):
        raise RuntimeError("Reloaded model predictions differ from the trained model.")

    metadata = {
        "model_version": "v3",
        "architecture": "TextVectorization -> Embedding -> Bidirectional(LSTM) -> Dropout -> Dense(sigmoid)",
        "threshold": threshold,
        "classes": LABELS,
        "positive_label": "spam",
        "output": "single sigmoid unit = spam probability",
        "input": "raw message string, shape (batch, 1)",
        "config": CONFIG,
        "vocabulary_size": model.get_layer("text_vectorization").vocabulary_size(),
        "standardize": "spam_v3>turkish_standardize (import train_bilstm before loading)",
        "class_weight": {LABELS[i]: w for i, w in class_weight.items()},
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "dataset": DATA_PATH.name,
        "dataset_fingerprint": split["fingerprint"],
        "feedback_file": FEEDBACK_PATH.name,
        "data_sources": data_stats,
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "split_indices_file": SPLIT_PATH.name,
        "random_state": RANDOM_STATE,
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "python_version": sys.version.split()[0],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics = {
        "threshold": threshold,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "validation_default_threshold": val_default,
        "validation": val_tuned,
        "test": test_metrics,
        "comparison_test_same_split": {
            "tfidf_lr_retrained": baseline_test,
            "v2_production": v2_test,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\nSaved {MODEL_PATH.name}, {METADATA_PATH.name}, {METRICS_PATH.name} to {MODEL_PATH.parent}/")
    print("The production model (app.py) was not changed.")


if __name__ == "__main__":
    main()
