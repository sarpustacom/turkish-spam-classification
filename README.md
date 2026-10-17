# Turkish SMS Spam Classifier

Classifies Turkish SMS messages as `spam` or `ham` and returns a spam probability.
The API serves a TensorFlow/Keras BiLSTM model (V3), with a simple web interface.
An earlier word + character TF-IDF + Logistic Regression model (scikit-learn) is kept as a backup.

Dataset: [`jason1966/aiwithcagri_trke-spam-sms-ve-sms-kategorileri`](https://huggingface.co/datasets/jason1966/aiwithcagri_trke-spam-sms-ve-sms-kategorileri)

## Project structure

```text
├── train.py          # data loading, cleaning, training, evaluation, export
├── train_bilstm.py   # V3 model used by the API (TensorFlow/Keras BiLSTM)
├── app.py            # Flask API + web UI route
├── templates/
│   └── index.html    # web UI page
├── static/
│   ├── style.css
│   └── script.js
├── data/             # Hugging Face dataset cache (created by train.py)
├── model/            # V3 (*_v3.*, served) and TF-IDF models (*.joblib, backup)
└── requirements.txt
```

## 1. Setup

```bash
conda create -n spam-classifier python=3.11 -y
conda activate spam-classifier
pip install -r requirements.txt
```

Run all commands below with the `spam-classifier` environment activated.

## 2. Train

```bash
python train.py
```

This downloads the dataset into `data/`, removes empty/invalid records and duplicates,
prints the class distribution, and splits the data 60/20/20 (stratified).
The spam decision threshold is chosen on the validation set (maximizing spam F1),
then the model is evaluated once on the test set.

Outputs in `model/`:

- `spam_classifier.joblib` – the full pipeline (both TF-IDF vectorizers + classifier)
- `metadata.json` – selected threshold, classes, split sizes, versions
- `metrics.json` – accuracy, precision, recall, F1 and confusion matrices

Load the model with the same scikit-learn version it was trained with
(recorded in `metadata.json`).

### V3 model (BiLSTM) — used by the API

```bash
python train_bilstm.py
```

Trains a TensorFlow/Keras model (TextVectorization → Embedding(128) → Bidirectional LSTM(64)
→ Dropout → Dense sigmoid) on `data/turkish_spam_combined.csv` (`message`, `label` columns),
with a stratified 60/20/20 split. The split indices are saved in `model/split_indices.json`
(with a dataset fingerprint) and reused on later runs. The threshold is chosen on validation data.

For comparison, the script also trains the V2 architecture (TF-IDF + Logistic Regression from
`train.py`) on the same split, and scores the production V2 model on the same test set.
V2 was trained on an unknown split, so its numbers may be optimistic.

Outputs: `model/spam_classifier_v3.keras`, `model/metadata_v3.json`, `model/metrics_v3.json`.
`app.py` serves this model with the threshold from `metadata_v3.json`. The V2 files
(`spam_classifier_v2.joblib`, `metadata.json`) are kept as a backup and are not used by the API.

The `.keras` file includes text preprocessing and takes raw strings. It uses a custom
Turkish-aware lowercasing function, so import `train_bilstm` before loading it:

```python
import keras, tensorflow as tf
import train_bilstm  # registers the custom standardize function
model = keras.models.load_model("model/spam_classifier_v3.keras")
model.predict(tf.constant([["Tebrikler! 100 TL kazandınız."]]))
```

## 3. Run the API

```bash
flask --app app run --port 5001 --debug
```

Or simply `python app.py`. The API runs at http://127.0.0.1:5001.
The model is loaded once at startup; the API never retrains it.
If the V3 model files are missing, the app refuses to start and asks you to run `train_bilstm.py`.
Startup takes a few seconds because TensorFlow loads the model.

Use `--debug` only for local development. To reach the API from another device
(e.g. an iPhone on the same network), add `--host 0.0.0.0`.

Port 5001 is used because macOS AirPlay Receiver already occupies port 5000.

## 4. Web interface

Open http://127.0.0.1:5001 in your browser after starting the API.

- Paste a message and click **Analiz Et**, or press Ctrl+Enter (Cmd+Enter on Mac).
- The result shows **SPAM TESPİT EDİLDİ** or **NORMAL MESAJ**, the spam probability, and a progress bar.
- Light/dark mode follows your system setting; the toggle in the header overrides it and is remembered in the browser.

The page is plain HTML, CSS and JavaScript (no frontend framework) and calls the same
`POST /api/v1/predict` endpoint described below. Messages are not stored or logged.

## 5. Test request (API)

```bash
curl -X POST http://127.0.0.1:5001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Tebrikler! 100 TL kazandınız."}'
```

Example response (values depend on your trained model):

```json
{"label": "spam", "spam_score": 0.87}
```

Health check:

```bash
curl http://127.0.0.1:5001/health
```

### Feedback (corrections)

In the web interface, **Analizi Düzelt** under a result lets the user pick the correct class
(Normal (Ham) / Spam) and save it. The same endpoint can be called directly:

```bash
curl -X POST http://127.0.0.1:5001/feedback \
  -H "Content-Type: application/json" \
  -d '{"message": "Tebrikler! 100 TL kazandınız.", "label": "spam"}'
```

```json
{"status": "saved", "label": "spam", "updated": false}
```

- Corrections are stored in `data/user_feedback.csv` (UTF-8, columns `message,label,updated_at`),
  created automatically. Sending the same message again updates its label (`"updated": true`).
- `label` must be exactly `ham` or `spam`; the message follows the same rules as `/api/v1/predict`.
- Messages starting with `= + - @ '` (or tab/CR) are stored with a leading `'` so spreadsheet apps
  don't run them as formulas. Remove one leading `'` when reading the file for training.
- Writes are locked (threads and processes) and atomic. A failed write returns HTTP 500.
- Feedback is never used for training automatically, and `data/` is git-ignored.

Invalid requests (missing body, non-string `message`, empty or whitespace-only
message, or more than 5000 characters) return HTTP 400 with an error message:

```json
{"error": "'message' must not be empty"}
```
