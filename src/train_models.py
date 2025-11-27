# src/train_models.py

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_PATH = BASE_DIR / "data" / "messages.csv"     # <-- using messages.csv
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")

# ---------------------------------------------------------------------
# Load dataset
#   Expected columns: one text column + one label column.
#   It will try common names automatically.
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

text_col_candidates = ["message", "text", "comment", "content"]
label_col_candidates = ["label", "target", "class"]

text_col = None
label_col = None

for c in text_col_candidates:
    if c in df.columns:
        text_col = c
        break

for c in label_col_candidates:
    if c in df.columns:
        label_col = c
        break

if text_col is None or label_col is None:
    raise ValueError(
        f"Could not find text/label columns.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Text candidates: {text_col_candidates}\n"
        f"Label candidates: {label_col_candidates}"
    )

print(f"Using text column:  {text_col}")
print(f"Using label column: {label_col}")

X_raw = df[text_col].astype(str)
y = df[label_col]

# ---------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(X_raw)

# ---------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

metrics = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    metrics[name] = {"accuracy": float(acc)}

    # Save each model
    filename = name.lower().replace(" ", "_") + "_model.joblib"
    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)
    print(f"Saved {name} to: {model_path}")

# ---------------------------------------------------------------------
# Save TF-IDF vectorizer + metrics
# ---------------------------------------------------------------------
tfidf_path = MODELS_DIR / "tfidf_vectorizer.joblib"
joblib.dump(tfidf, tfidf_path)
print(f"\nSaved TF-IDF vectorizer to: {tfidf_path}")

metrics_path = MODELS_DIR / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)
print(f"Saved metrics to: {metrics_path}")

print("\n✅ All models and vectorizer saved successfully!")
