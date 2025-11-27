from pathlib import Path
import json
import joblib
import traceback
import streamlit as st
from typing import Tuple, Dict, Any

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🚨", layout="wide")

# Uploaded assets (optional)
HERO_IMAGE = "/mnt/data/632a80c9-d4a6-4407-b72f-c033c6a660ad.png"
RESULT_SAMPLE = "/mnt/data/292af1ad-27d1-41ac-9ec7-bbe24c8ecc7d.png"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ----------------------------
# Styles (aim: match screenshot exactly)
# ----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');

    :root{--nav-height:60px}
    html, body, .stApp {
        height:100%;
        background:#f6f6f6;
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        color:#111;
    }

    /* Top navbar */
    .top-nav{height:var(--nav-height);background:#24313b;display:flex;align-items:center;justify-content:space-between;padding:10px 48px;color:#fff;border-radius:15px}
    .brand{font-family:'Bebas Neue', cursive;font-size:22px}
    .nav-links a{color:#dfe7ea;text-decoration:none;margin-left:22px;font-size:14px}

    /* Hero card */
    .hero{max-width:1200px;margin:36px auto;padding:64px 80px;border-radius:50px;background:linear-gradient(180deg,#6a1b9a 0%, #6036a6 28%, #4b4f36 100%);box-shadow:0 8px 20px rgba(0,0,0,0.12);color:#000;position:relative}
    .hero h1{font-family:'Bebas Neue', cursive;font-size:56px;text-align:center;margin:0 0 6px}
    .hero p.lead{font-size:16px;text-align:center;margin:0 0 28px;color:rgba(0,0,0,0.65)}

    /* Input container placement */
    .hero .input-wrap{display:flex;flex-direction:column;align-items:center;gap:18px}

    /* Style the Streamlit text_area to match screenshot */
    .stTextArea textarea{background:#fff !important;color:#111 !important;border-radius:8px !important;padding:18px !important;box-shadow:none !important;border:0 !important;max-width:820px !important;width:100% !important;height:56px !important}
    .stTextArea > label {display:none}

    /* Detect button -> black wide */
    .stButton>button{background:#000 !important;color:#fff !important;padding:14px 28px !important;border-radius:6px !important;border:0 !important;font-weight:700;width:420px !important}

    /* Result page */
    .result-box{max-width:800px;margin:10px auto;padding:10px;border-radius:28px;background:#fff2f2;border:3px solid #d9534f;text-align:center}
    .result-box h2{margin:0;font-size:28px}
    .model-card{max-width:300px;margin:8px 50px;padding:5px;border-radius:25px;background:#f6f6f6;box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);text-align:center}
    .model-card h4{margin:0 0 4px 0}

    /* Back button */
    .back-btn > button{background:#2b9be0 !important;color:#fff !important;padding:10px 18px !important;border-radius:6px !important;border:0 !important}

    /* Responsive */
    @media (max-width:900px){
        .hero{padding:36px 22px}
        .stButton>button{width:90% !important}
        .stTextArea textarea{max-width:100% !important}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Top nav
# ----------------------------
st.markdown(
    """
    <div class="top-nav">
        <div class="brand">Cyberbullying Detector</div>
        <div class="nav-links">
            <a href="#">Main Page</a>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers: load models + predict
# ----------------------------

def load_models_and_vectorizer() -> Tuple[Dict[str, Any], Any, Dict[str, Any], str]:
    """Load joblib models and tfidf vectorizer from the models folder.
    Returns (models_dict, tfidf, metrics_dict, error_text)
    If an error occurs, models_dict/tfidf/metrics will be empty/None and error_text will contain a traceback.
    """
    models = {}
    tfidf = None
    metrics = {}
    error_text = ""

    try:
        # Expecting joblib files in MODELS_DIR
        if not MODELS_DIR.exists():
            raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

        files_expected = {
            "Logistic Regression": MODELS_DIR / "logistic_regression_model.joblib",
            "Decision Tree": MODELS_DIR / "decision_tree_model.joblib",
            "Random Forest": MODELS_DIR / "random_forest_model.joblib",
            "TFIDF": MODELS_DIR / "tfidf_vectorizer.joblib",
        }

        # Load each model if present
        for name, path in files_expected.items():
            if not path.exists():
                # skip missing models but note it in metrics
                continue
            if name == "TFIDF":
                tfidf = joblib.load(str(path))
            else:
                models[name] = joblib.load(str(path))

        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

    except Exception:
        error_text = traceback.format_exc()

    return models, tfidf, metrics, error_text


def predict_message(model, tfidf, message: str):
    if model is None or tfidf is None:
        return None, None
    X = tfidf.transform([message])
    pred = None
    score = None
    try:
        pred = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            # handle binary case and multiclass safely
            if len(proba) == 2:
                score = float(proba[1])
            else:
                score = float(max(proba))
    except Exception:
        # model might not support predict_proba or there's a shape problem
        score = None
    return pred, score

# ----------------------------
# Session state and flow
# ----------------------------
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

models, tfidf, metrics, load_error = load_models_and_vectorizer()

# If there was an error loading models/vectorizer, show it clearly so user can debug
if load_error:
    st.error("Error while loading models/vectorizer. See details below.")
    st.code(load_error)

# ----------------------------
# Home (hero) layout — pixel-perfect
# ----------------------------

def show_home():
    st.markdown(
        f"""
        <div class="hero">
            <h1>Cyberbullying Detector</h1>
            <p class="lead">Detecting cyberbullying in a simple way</p>
            <div class="input-wrap"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        user_input = st.text_area(
            label="",
            value=st.session_state.get("last_input", ""),
            height=56,
            placeholder="Example: You are so stupid, nobody likes you...",
            key="user_textarea",
        )
        if st.button("Detect"):
            if not user_input or not user_input.strip():
                st.warning("Please enter a message before analyzing.")
            else:
                st.session_state.last_input = user_input
                st.session_state.show_results = True
                # prefer not to force-restart the script; let the control flow show results on next render
                st.experimental_rerun()

    st.markdown("---")
    st.info("This interface reproduces the demonstration UI with three model predictions shown on the results screen.")

# ----------------------------
# Results layout to match screenshot
# ----------------------------

def show_results():
    # compute per-model predictions + confidences
    confidences = []

    # If there are no models or no tfidf, show helpful message + sample image
    if not models or tfidf is None:
        st.markdown('<div class="result-box"><h2>Models unavailable</h2><p style="font-size:18px">Place trained models and TF-IDF in the <code>models/</code> folder to enable predictions.</p></div>', unsafe_allow_html=True)
        if Path(RESULT_SAMPLE).exists():
            st.image(RESULT_SAMPLE, caption="Result sample layout", use_column_width=True)
        else:
            st.info("Result sample image not found; expected path: {}".format(RESULT_SAMPLE))

        if st.button("Back to Home"):
            st.session_state.show_results = False
            st.experimental_rerun()
        return

    # If we have models + tfidf, compute predictions
    st.markdown('<div class="result-box"><h2>Cyberbullying Detected!</h2></div>', unsafe_allow_html=True)
    st.markdown("### Individual Model Predictions:")

    for name, model in models.items():
        pred, score = predict_message(model, tfidf, st.session_state.last_input)
        if score is not None:
            confidences.append(score)

        # interpret prediction
        label = "Unknown"
        if pred is None:
            label = "Model error"
        else:
            p = str(pred).lower()
            if p in ["1", "inappropriate", "toxic", "abusive", "bullying", "yes"]:
                label = "Bullying"
            else:
                label = "Not Bullying"

        st.markdown(
            f"""
            <div class="model-card">
                <h4>{name}</h4>
                <div style="font-size:14px;">
                    <strong>Prediction:</strong> {label}<br />
                    <strong>Confidence:</strong> {('N/A' if score is None else f'{score*100:.2f}%')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # show aggregated confidence if available
    if confidences:
        agg = sum(confidences) / len(confidences)
        st.markdown(f"**Aggregated confidence (mean of models):** {agg*100:.2f}%")

    if st.button("Back to Home"):
        st.session_state.show_results = False
        st.experimental_rerun()

# ----------------------------
# Run app
# ----------------------------
if not st.session_state.show_results:
    show_home()
else:
    show_results()
