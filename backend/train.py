"""Train + evaluate three models and save them to models/.

1. Naive Bayes (from scratch)       -> Bayes theorem
2. Naive Bayes (scikit-learn)       -> baseline comparison, TF-IDF/count features
3. Neural network (deep learning)   -> multi-layer perceptron on TF-IDF features

Usage:  python train.py [path/to/dataset.csv]
Dataset needs columns 'label' (ham/spam) and 'text'. The Kaggle 'spam.csv' (v1/v2) also works.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from naive_bayes import NaiveBayesSpam
from nlp_utils import preprocess_to_string

MODELS = Path("models")
MODELS.mkdir(exist_ok=True)


def load(path):
    df = pd.read_csv(path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):           # Kaggle format
        df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].str.lower().str.strip()
    return df[df["label"].isin(["ham", "spam"])]


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label="spam", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, pos_label="spam", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, pos_label="spam", zero_division=0), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/emails.csv"
    df = load(path)
    print(f"Loaded {len(df)} rows from {path}  (spam={int((df.label == 'spam').sum())})")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

    results = {}

    # 1) from-scratch Bayes
    nb = NaiveBayesSpam(alpha=1.0).fit(X_train, y_train)
    results["Naive Bayes (scratch)"] = metrics(y_test, [nb.predict(t) for t in X_test])
    joblib.dump(nb, MODELS / "nb_scratch.joblib")

    # 2) sklearn Bayes
    clean_train = X_train.map(preprocess_to_string)
    clean_test = X_test.map(preprocess_to_string)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
    sk_nb = MultinomialNB(alpha=0.5).fit(cv.fit_transform(clean_train), y_train)
    results["Naive Bayes (sklearn)"] = metrics(y_test, sk_nb.predict(cv.transform(clean_test)))

    # 3) Deep learning: MLP with two hidden layers
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", early_stopping=True,
                        max_iter=200, random_state=42)
    mlp.fit(tfidf.fit_transform(clean_train), y_train)
    results["Neural Network (MLP)"] = metrics(y_test, mlp.predict(tfidf.transform(clean_test)))
    joblib.dump({"tfidf": tfidf, "model": mlp}, MODELS / "mlp.joblib")

    for name, m in results.items():
        print(f"{name:24s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f}")
    (MODELS / "metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved models to", MODELS.resolve())


if __name__ == "__main__":
    main()
