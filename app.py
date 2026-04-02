"""
=============================================================
  app.py
  Fake Website Detection System
  Role: Streamlit web UI — enter URL → get prediction
  Run: streamlit run app.py
=============================================================
"""

import os, pickle, sys
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ── Make sibling modules importable regardless of cwd ───────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_features

# ─────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishGuard — Fake Website Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — cyberpunk / dark-terminal aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

/* ── Root variables ── */
:root {
    --bg:       #0b0d12;
    --card:     #12151f;
    --border:   #1e2535;
    --cyan:     #00e5ff;
    --red:      #ff1744;
    --green:    #00e676;
    --gold:     #ffd600;
    --text:     #cdd6f4;
    --muted:    #6c7a9c;
    --radius:   12px;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Exo 2', sans-serif;
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 780px; }

/* ── Title banner ── */
.banner {
    background: linear-gradient(135deg, #0f1724 0%, #0b0d12 60%, #0f1a1f 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--cyan);
    border-radius: var(--radius);
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 70% 60% at 50% -10%, rgba(0,229,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.banner h1 {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.5px;
    color: #fff;
    margin: 0 0 .25rem;
}
.banner h1 span { color: var(--cyan); }
.banner p { color: var(--muted); font-size: .9rem; margin: 0; font-family: var(--mono); }

/* ── Input styling ── */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: .95rem !important;
    padding: .6rem 1rem !important;
    transition: border-color .2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,.12) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00b8cc 0%, #005f6b 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: .6rem 2.5rem !important;
    cursor: pointer !important;
    transition: opacity .2s, transform .1s !important;
    letter-spacing: .5px;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

/* ── Result cards ── */
.result-card {
    border-radius: var(--radius);
    padding: 1.5rem 1.8rem;
    margin: 1.2rem 0;
    border-left: 5px solid;
    position: relative;
    overflow: hidden;
}
.result-card.legit {
    background: linear-gradient(135deg, rgba(0,230,118,.08), rgba(0,0,0,0));
    border-color: var(--green);
}
.result-card.fake {
    background: linear-gradient(135deg, rgba(255,23,68,.10), rgba(0,0,0,0));
    border-color: var(--red);
}
.result-icon { font-size: 2.5rem; margin-bottom: .4rem; }
.result-label {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 1.7rem;
    margin: 0;
}
.result-label.legit { color: var(--green); }
.result-label.fake  { color: var(--red); }
.result-sub { color: var(--muted); font-family: var(--mono); font-size: .83rem; margin-top: .2rem; }

/* ── Confidence bar ── */
.conf-wrap { margin-top: 1rem; }
.conf-label { font-size: .82rem; color: var(--muted); font-family: var(--mono); margin-bottom: .3rem; }
.conf-bar-bg {
    background: var(--border);
    border-radius: 20px;
    height: 10px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width .8s ease;
}
.conf-pct {
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: .4rem;
}

/* ── Feature pill grid ── */
.feat-grid { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .8rem; }
.feat-pill {
    font-family: var(--mono);
    font-size: .77rem;
    padding: .25rem .7rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--card);
    white-space: nowrap;
}
.feat-pill.warn { border-color: var(--red);  color: var(--red);  background: rgba(255,23,68,.08); }
.feat-pill.ok   { border-color: var(--cyan); color: var(--cyan); background: rgba(0,229,255,.06); }
.feat-pill.neu  { border-color: var(--gold); color: var(--gold); background: rgba(255,214,0,.06); }

/* ── Section headings ── */
.section-head {
    font-family: var(--mono);
    font-size: .78rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 1.4rem 0 .6rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: .4rem;
}

/* ── Metric chips (sidebar / expander) ── */
.metric-row { display: flex; gap: .7rem; flex-wrap: wrap; margin-top: .5rem; }
.metric-chip {
    font-family: var(--mono);
    font-size: .8rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .3rem .8rem;
    color: var(--text);
}
.metric-chip strong { color: var(--cyan); }

/* ── Examples list ── */
.ex-url {
    font-family: var(--mono);
    font-size: .82rem;
    color: var(--cyan);
    cursor: pointer;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .3rem .8rem;
    margin-bottom: .3rem;
    display: inline-block;
    text-decoration: none;
    transition: border-color .15s;
}
.ex-url:hover { border-color: var(--cyan); }

/* ── Streamlit image caption ── */
.css-1kyxreq { color: var(--muted) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: .85rem !important;
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load model & meta (with auto-train if missing)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄  Loading model …")
def load_model():
    model_path = "models/best_model.pkl"
    meta_path  = "models/model_meta.pkl"
    scaler_path= "models/scaler.pkl"

    if not (os.path.exists(model_path) and
            os.path.exists(meta_path)  and
            os.path.exists(scaler_path)):
        # Auto-train on first run
        from model_training import train
        train()

    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(meta_path,   "rb") as f: meta   = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    return model, scaler, meta


model, scaler, meta = load_model()
feature_names  = meta["feature_names"]
model_name     = meta["model_name"]
model_metrics  = meta["metrics"]


# ─────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────
def predict_url(url: str):
    """
    Extract features → scale → predict.
    Returns: label (str), confidence (float 0-1), feature dict
    """
    feats = extract_features(url, dns_check=False)

    # Build feature vector in the same order as training
    feat_vec = np.array([feats.get(f, 0) for f in feature_names]).reshape(1, -1)
    feat_scaled = scaler.transform(feat_vec)

    proba = model.predict_proba(feat_scaled)[0]   # [P(fake), P(legit)]
    
    legit_score = proba[1]
    is_legit = legit_score > 0.80
    
    label = "Legitimate URL" if is_legit else "Fake URL"
    confidence = legit_score if is_legit else proba[0]
    
    return label, confidence, feats, proba, feat_scaled[0]


# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────
def render_result(label: str, confidence: float, feats: dict, url: str):
    is_legit = (label == "Legitimate URL")
    card_cls  = "legit" if is_legit else "fake"
    icon      = "✅" if is_legit else "❌"
    bar_color = "#00e676" if is_legit else "#ff1744"
    pct_color = "#00e676" if is_legit else "#ff1744"
    pct       = confidence * 100

    st.markdown(f"""
    <div class="result-card {card_cls}">
      <div class="result-icon">{icon}</div>
      <p class="result-label {card_cls}">{label}</p>
      <p class="result-sub">{url[:72]}{"…" if len(url)>72 else ""}</p>
      <div class="conf-wrap">
        <div class="conf-label">CONFIDENCE SCORE</div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div>
        </div>
        <div class="conf-pct" style="color:{pct_color};">{pct:.1f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature pills ─────────────────────────────────────
    st.markdown('<div class="section-head">Extracted Signals</div>', unsafe_allow_html=True)

    WARN_RULES = {
        "has_https":               lambda v: v == 0,
        "has_ip_address":          lambda v: v == 1,
        "has_suspicious_tld":      lambda v: v == 1,
        "num_at_signs":            lambda v: v > 0,
        "num_brand_keywords":      lambda v: v > 0,
        "has_prefix_suffix_hyphen":lambda v: v == 1,
        "url_length":              lambda v: v > 75,
        "num_dots":                lambda v: v > 4,
        "num_percent":             lambda v: v > 2,
    }

    pills_html = '<div class="feat-grid">'
    for fname, fval in feats.items():
        if fname == "dns_resolves":
            continue
        is_warn = WARN_RULES.get(fname, lambda v: False)(fval)
        cls     = "warn" if is_warn else ("ok" if fval not in (0, -1) else "neu")
        label_txt = fname.replace("_", " ")
        pills_html += f'<span class="feat-pill {cls}">{label_txt}: <b>{fval}</b></span>'
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)


def render_model_info():
    st.markdown('<div class="section-head">Model Info</div>', unsafe_allow_html=True)
    row = '<div class="metric-row">'
    for k, v in model_metrics.items():
        row += f'<span class="metric-chip"><strong>{k}</strong>: {v:.4f}</span>'
    row += f'<span class="metric-chip"><strong>Algorithm</strong>: {model_name}</span>'
    row += "</div>"
    st.markdown(row, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
# Banner
st.markdown("""
<div class="banner">
  <h1>🛡️ &nbsp;Phish<span>Guard</span></h1>
  <p>ML-powered fake website & phishing URL detector · Python + scikit-learn</p>
</div>
""", unsafe_allow_html=True)

# ── Input row ────────────────────────────────────────────────
url_input = st.text_input(
    label="",
    placeholder="https://enter-any-url-to-check.com",
    key="url_box",
    label_visibility="collapsed",
)
check_btn = st.button("🔍  Check Website", use_container_width=False)

# ── Quick examples ───────────────────────────────────────────
with st.expander("💡  Try example URLs"):
    st.markdown("""
    **✅ Legitimate**
    ```
    https://www.google.com
    https://www.github.com
    https://www.amazon.com
    ```
    **🚨 Phishing / Fake**
    ```
    http://paypa1.com-secure-login.tk/account
    http://192.168.1.1/bank-login
    http://apple-id-locked.com/verify-now
    http://secure-ebay-account.com/signin
    ```
    """)

# ── Run prediction ───────────────────────────────────────────
if check_btn:
    raw = url_input.strip()
    if not raw:
        st.warning("⚠️  Please enter a URL first.")
    else:
        with st.spinner("Analysing URL …"):
            try:
                label, confidence, feats, proba, feat_scaled_1d = predict_url(raw)
                render_result(label, confidence, feats, raw)
                
                # ── Dynamic Model metrics per URL ────────────────────────────
                render_model_info()

                # ── Dynamic URL Report Charts ───────────────────────────────
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.markdown('<div class="section-head">Feature Importance (For this URL)</div>', unsafe_allow_html=True)
                    if hasattr(model, "feature_importances_"):
                        impact = model.feature_importances_ * feat_scaled_1d
                    elif hasattr(model, "coef_"):
                        impact = model.coef_[0] * feat_scaled_1d
                    else:
                        impact = feat_scaled_1d
                    # Sort features by absolute impact
                    imp_df = pd.DataFrame({"Impact": impact}, index=feature_names)
                    imp_df["Abs"] = imp_df["Impact"].abs()
                    top_imp = imp_df.sort_values(by="Abs", ascending=False).head(10)
                    st.bar_chart(top_imp["Impact"])

                with img_col2:
                    st.markdown('<div class="section-head">Model Comparison (Probabilities)</div>', unsafe_allow_html=True)
                    prob_df = pd.DataFrame({"Probability": [proba[0], proba[1]]}, index=["Fake", "Legitimate"])
                    st.bar_chart(prob_df)
                    
            except Exception as e:
                st.error(f"❌  Error analysing URL: {e}")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #1e2535;margin:2rem 0 .8rem;">
<p style="color:#3d4a60;font-size:.75rem;font-family:'Share Tech Mono',monospace;text-align:center;">
PhishGuard · Built with Python, scikit-learn & Streamlit ·
For educational / research purposes only.
</p>
""", unsafe_allow_html=True)
