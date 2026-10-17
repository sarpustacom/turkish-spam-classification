"""Train a Turkish SMS spam classifier (word + char TF-IDF + Logistic Regression)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

DATASET_NAME = "jason1966/aiwithcagri_trke-spam-sms-ve-sms-kategorileri"
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"
RANDOM_STATE = 42
LABELS = ["ham", "spam"]


def load_data() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, cache_dir=str(DATA_DIR))
    # Merge all available splits; we create our own splits below.
    df = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)

    missing = {"metin", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df[["metin", "label"]]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_raw = len(df)
    df = df.dropna(subset=["metin", "label"]).copy()
    df["metin"] = df["metin"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["metin"] != ""]
    df = df[df["label"].isin(LABELS)]
    n_valid = len(df)

    df = df.drop_duplicates(subset="metin").reset_index(drop=True)

    print(f"Raw records:        {n_raw}")
    print(f"Invalid/empty:      {n_raw - n_valid}")
    print(f"Duplicates removed: {n_valid - len(df)}")
    print(f"Final records:      {len(df)}")
    return df


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def spam_probabilities(model: Pipeline, texts: pd.Series) -> np.ndarray:
    spam_index = list(model.classes_).index("spam")
    return model.predict_proba(texts)[:, spam_index]


def select_threshold(y_true: pd.Series, spam_proba: np.ndarray) -> float:
    """Return the threshold that maximizes spam F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, spam_proba, pos_label="spam"
    )
    # The last precision/recall pair has no threshold.
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(y_true: pd.Series, spam_proba: np.ndarray, threshold: float) -> dict:
    y_pred = np.where(spam_proba >= threshold, "spam", "ham")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "spam_precision": precision_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "spam_recall": recall_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "spam_f1": f1_score(y_true, y_pred, pos_label="spam", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
        "report": classification_report(
            y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True
        ),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n=== {name} ===")
    for key in ("accuracy", "spam_precision", "spam_recall", "spam_f1", "macro_f1"):
        print(f"{key:15s} {metrics[key]:.4f}")
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]["matrix"]
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"           ham   spam")
    print(f"  ham   {tn:6d} {fp:6d}")
    print(f"  spam  {fn:6d} {tp:6d}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = clean_data(load_data())

    counts = df["label"].value_counts()
    print("\nClass distribution:")
    for label, count in counts.items():
        print(f"  {label:5s} {count:6d} ({count / len(df):.2%})")
    if set(counts.index) != set(LABELS):
        raise ValueError("Both 'spam' and 'ham' classes are required for training.")

    # 60/20/20 stratified split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["metin"], df["label"], test_size=0.4, stratify=df["label"], random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print("\nTraining...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    val_proba = spam_probabilities(model, X_val)
    val_default = evaluate(y_val, val_proba, 0.5)
    threshold = select_threshold(y_val, val_proba)
    val_tuned = evaluate(y_val, val_proba, threshold)
    print(f"\nSelected threshold (max validation spam F1): {threshold:.4f}")
    print_metrics("Validation @ 0.5", val_default)
    print_metrics(f"Validation @ {threshold:.4f}", val_tuned)

    # Test set is evaluated exactly once, with the threshold chosen on validation.
    test_metrics = evaluate(y_test, spam_probabilities(model, X_test), threshold)
    print_metrics(f"Test @ {threshold:.4f}", test_metrics)

    joblib.dump(model, MODEL_DIR / "spam_classifier.joblib")

    metadata = {
        "threshold": threshold,
        "classes": list(model.classes_),
        "positive_label": "spam",
        "dataset": DATASET_NAME,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics = {
        "threshold": threshold,
        "validation_default_threshold": val_default,
        "validation": val_tuned,
        "test": test_metrics,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\nSaved model and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
