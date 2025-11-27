"""
Robust training + save script for the Inappropriate Message Detection project.

What it does:
- Loads data/data/messages.csv
- Auto-detects the text column (common names: message, text, msg, content, tweet)
- Auto-detects the label column (common names: label, target, class, category)
- If labels are not numeric and there are exactly 2 unique values, maps them to 0/1
- Trains a TF-IDF vectorizer + LogisticRegression model (fast baseline)
- Saves artifacts to the models/ folder:
    - models/tfidf_vectorizer.joblib
    - models/best_model.joblib

Usage (from project root):
    python src/train_models.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# ---------- Configuration ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "messages.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Column name candidates (common possibilities)
TEXT_CANDIDATES = ["message", "text", "msg", "content", "tweet", "comment", "body"]
LABEL_CANDIDATES = ["label", "target", "class", "category", "y", "label_id", "is_inappropriate", "unsafe"]

# ---------- Helpers ----------
def choose_column(columns, candidates):
    """Return first column from candidates that exists in columns (case-insensitive)."""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# ---------- Load data ----------
print("Loading dataset:", DATA_PATH)
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}\nPlease create the file and ensure it has text + label columns.")

df = pd.read_csv(DATA_PATH)
if df.shape[0] == 0:
    raise SystemExit("Dataset appears empty (0 rows). Please check data/messages.csv.")

print("Columns found in CSV:", list(df.columns))

# Detect text and label columns
text_col = choose_column(df.columns, TEXT_CANDIDATES)
label_col = choose_column(df.columns, LABEL_CANDIDATES)

# If auto-detection failed, be helpful and try some heuristics
if text_col is None:
    # choose the first column that is string/object dtype
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if len(obj_cols) == 1:
        text_col = obj_cols[0]
        print(f"[heuristic] Using only object column '{text_col}' as text column.")
    elif len(obj_cols) > 1:
        print("Unable to auto-detect text column. Object (string) columns found:", obj_cols)
        print("Please rename the text column to one of:", TEXT_CANDIDATES)
        sys.exit(1)
    else:
        print("No obvious text column found. Please ensure your CSV has a text column.")
        sys.exit(1)

if label_col is None:
    # try to find numeric columns with small number of unique values (likely labels)
    numeric_candidates = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c])]
    chosen = None
    for c in numeric_candidates:
        if df[c].nunique() <= 10:  # heuristic
            chosen = c
            break
    if chosen:
        label_col = chosen
        print(f"[heuristic] Using numeric-ish column '{label_col}' as label column.")
    else:
        # last resort: if there's one non-text column, use it
        non_text_cols = [c for c in df.columns if c != text_col]
        if len(non_text_cols) == 1:
            label_col = non_text_cols[0]
            print(f"[heuristic] Using only non-text column '{label_col}' as label column.")
        else:
            print("Unable to auto-detect label column. Columns available:", df.columns.tolist())
            print("Please rename the label column to one of:", LABEL_CANDIDATES)
            sys.exit(1)

print(f"Using text column: '{text_col}'")
print(f"Using label column: '{label_col}'")

# Extract X, y
X = df[text_col].astype(str).values
y_raw = df[label_col]

# Handle label encoding
# If numeric and only two unique values, map to 0/1
unique_vals = pd.Series(y_raw).dropna().unique()
n_unique = len(unique_vals)
print(f"Label cardinality: {n_unique} unique values -> {list(unique_vals)[:10]}")

# If labels are strings or not 0/1 integers, encode with LabelEncoder if binary; else require binary for this pipeline.
label_encoder = None
if n_unique == 0:
    raise SystemExit("Label column appears empty. Please check your CSV.")
elif n_unique == 1:
    print("Only 1 unique label found - nothing to train. Exiting.")
    sys.exit(1)
elif n_unique == 2:
    # safe to encode to 0/1
    if pd.api.types.is_numeric_dtype(y_raw):
        # map min -> 0, max -> 1 if not already 0/1
        uniq_sorted = sorted(set(y_raw))
        if set(uniq_sorted) == {0, 1}:
            y = y_raw.astype(int).values
            print("Labels already 0/1 numeric.")
        else:
            low, high = uniq_sorted[0], uniq_sorted[-1]
            print(f"Mapping numeric labels {low} -> 0 and {high} -> 1.")
            y = y_raw.apply(lambda v: 1 if v == high else 0).astype(int).values
    else:
        # string labels: use LabelEncoder (mapping printed)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str).values)
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print("LabelEncoder mapping:", mapping)
else:
    # More than 2 classes found
    print("Found more than 2 unique labels. This script expects a binary classification (appropriate vs inappropriate).")
    print("If your task is multi-class you can modify the script to train a multi-class model.")
    print("Found labels (first 20):", list(unique_vals)[:20])
    sys.exit(1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Vectorizer + Model
print("Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))

print("Fitting vectorizer on training data...")
X_train_tfidf = vectorizer.fit_transform(X_train)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train_tfidf, y_train)

# Quick eval
train_score = model.score(X_train_tfidf, y_train)
X_test_tfidf = vectorizer.transform(X_test)
test_score = model.score(X_test_tfidf, y_test)
print(f"Train accuracy: {train_score:.4f}  Test accuracy: {test_score:.4f}")

# Save artifacts
vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
model_path = os.path.join(MODELS_DIR, "best_model.joblib")
enc_path = os.path.join(MODELS_DIR, "label_encoder.joblib") if label_encoder is not None else None

joblib.dump(vectorizer, vec_path)
joblib.dump(model, model_path)
if label_encoder is not None:
    joblib.dump(label_encoder, enc_path)

print("Saved vectorizer ->", vec_path)
print("Saved model ->", model_path)
if enc_path:
    print("Saved label encoder ->", enc_path)
print("All done.")
