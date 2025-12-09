import os
import re
import pickle
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup

# ============================================
# ONNX RUNTIME (REPLACES TENSORFLOW)
# ============================================
import onnxruntime as ort

# ============================================
# CONFIG
# ============================================
MAX_SEQ_LEN = 50
NUM_WORDS = 20000

TFIDF_PATH = "tfidf_vectorizer.pkl"
LOGREG_PATH = "log_reg.pkl"
TOKENIZER_PATH = "tokenizer.pkl"
ONNX_MODEL_PATH = "cnn_model.onnx"   # <-- USE ONNX MODEL

LABEL_MAP = {0: "Real", 1: "Fake"}

# ============================================
# CACHED LOADERS
# ============================================
@st.cache_resource
def load_tfidf_and_logreg():
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(LOGREG_PATH, "rb") as f:
        logreg = pickle.load(f)
    return tfidf, logreg


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_onnx_model():
    """Loads the ONNX CNN model."""
    return ort.InferenceSession(ONNX_MODEL_PATH)


# ============================================
# CLEAN TEXT
# ============================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================
# TOKENIZER → SEQUENCE
# (NO TENSORFLOW – custom padding)
# ============================================
def text_to_sequence(text, tok_dict):
    cleaned = clean_text(text)
    word_index = tok_dict["word_index"]
    num_words = tok_dict.get("num_words", NUM_WORDS)

    seq = []
    for w in cleaned.split():
        idx = word_index.get(w)
        if idx and idx < num_words:
            seq.append(idx)

    # If empty sequence, add placeholder
    if not seq:
        seq = [0]

    # Manual padding to MAX_SEQ_LEN
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    else:
        seq = seq + [0] * (MAX_SEQ_LEN - len(seq))

    return np.array(seq, dtype=np.int32).reshape(1, MAX_SEQ_LEN)


# ============================================
# PREDICT FUNCTIONS
# ============================================
def decode_label(y_int: int) -> str:
    return LABEL_MAP.get(y_int, "Unknown")


def predict_with_logreg(text: str):
    tfidf, logreg = load_tfidf_and_logreg()
    X = tfidf.transform([clean_text(text)])
    prob = logreg.predict_proba(X)[0, 1]
    return decode_label(int(prob >= 0.5)), float(prob)


def predict_with_cnn(text: str):
    tok_dict = load_tokenizer()
    sess = load_onnx_model()

    seq = text_to_sequence(text, tok_dict)

    # ONNX input name
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: seq})

    prob = float(output[0][0][0])
    return decode_label(int(prob >= 0.5)), prob


# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="Fake News Detection", layout="wide")

st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117!important; color:white!important; }
        textarea, input { background-color:#1b1f24!important; color:white!important; }
        </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
input_mode = st.sidebar.radio("📝 Input Mode:", ["Write text", "Paste URL"])
st.sidebar.markdown("---")
st.sidebar.caption("Created by Bhavya Pandya 🚀")

st.title("📰 Fake News Detection")
st.caption("Logistic Regression + ONNX CNN (TensorFlow-Free Deployment)")

user_text = ""

if input_mode == "Write text":
    user_text = st.text_area("✍️ Enter your news text here:", height=200)

else:
    url = st.text_input("🔗 Paste article URL:")
    if url:
        try:
            with st.spinner("Fetching article..."):
                html = requests.get(url, timeout=10).text
                soup = BeautifulSoup(html, "html.parser")
                user_text = "\n".join(p.get_text() for p in soup.find_all("p"))
            st.success("Article loaded!")
            st.text_area("Extracted text:", user_text[:2000], height=200)
        except Exception as e:
            st.error(f"Error fetching article: {e}")


# ============================================
# BUTTONS
# ============================================
col1, col2 = st.columns(2)
lr_result = None
cnn_result = None

with col1:
    if st.button("🔎 Predict with Logistic Regression"):
        if user_text.strip():
            lr_result = predict_with_logreg(user_text)
        else:
            st.warning("Enter text first.")

with col2:
    if st.button("🧠 Predict with CNN (ONNX)"):
        if user_text.strip():
            cnn_result = predict_with_cnn(user_text)
        else:
            st.warning("Enter text first.")


# ============================================
# RESULTS
# ============================================
st.markdown("---")

if lr_result:
    label, prob = lr_result
    st.subheader("📌 Logistic Regression Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability Fake:** {prob:.3f}")

if cnn_result:
    label, prob = cnn_result
    st.subheader("🧠 CNN (ONNX) Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability Fake:** {prob:.3f}")

st.markdown("---")
st.caption("*Educational tool — always verify news from reliable sources.*")
