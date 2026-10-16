"""Flask API:  POST /api/predict {text}  ->  verdicts from Bayes + neural network."""
import json
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from nlp_utils import preprocess_to_string

app = Flask(__name__)
MODELS = Path(__file__).parent / "models"

nb = joblib.load(MODELS / "nb_scratch.joblib")
mlp_bundle = joblib.load(MODELS / "mlp.joblib")
METRICS = json.loads((MODELS / "metrics.json").read_text())


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.post("/api/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Please provide email text."), 400

    p = nb.predict_proba(text)
    mlp_p = mlp_bundle["model"].predict_proba(mlp_bundle["tfidf"].transform([preprocess_to_string(text)]))[0]
    mlp_spam = float(mlp_p[list(mlp_bundle["model"].classes_).index("spam")])

    combined = (p["spam"] + mlp_spam) / 2
    return jsonify(
        verdict="spam" if combined >= 0.5 else "ham",
        spam_score=round(combined, 4),
        models={
            "naive_bayes": {"spam": round(p["spam"], 4), "ham": round(p["ham"], 4)},
            "neural_network": {"spam": round(mlp_spam, 4), "ham": round(1 - mlp_spam, 4)},
        },
        explanation=nb.explain(text),
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
