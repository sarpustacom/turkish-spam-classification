# Turkish SMS Spam Classifier

Classifies Turkish SMS messages as `spam` or `ham` and returns a spam probability.
Word + character TF-IDF features with Logistic Regression (scikit-learn), served with Flask, with a simple web interface.

Dataset: [`jason1966/aiwithcagri_trke-spam-sms-ve-sms-kategorileri`](https://huggingface.co/datasets/jason1966/aiwithcagri_trke-spam-sms-ve-sms-kategorileri)

## Project structure

```text
├── train.py          # data loading, cleaning, training, evaluation, export
├── app.py            # Flask API + web UI route
├── templates/
│   └── index.html    # web UI page
├── static/
│   ├── style.css
│   └── script.js
├── data/             # Hugging Face dataset cache (created by train.py)
├── model/            # spam_classifier.joblib, metadata.json, metrics.json
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

## 3. Run the API

```bash
flask --app app run --port 5001 --debug
```

Or simply `python app.py`. The API runs at http://127.0.0.1:5001.
The model is loaded once at startup; the API never retrains it.
If the `model/` files are missing, the app refuses to start and asks you to run `train.py`.

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

Invalid requests (missing body, non-string `message`, empty or whitespace-only
message, or more than 5000 characters) return HTTP 400 with an error message:

```json
{"error": "'message' must not be empty"}
```
