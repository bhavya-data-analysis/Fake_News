import os
import re
import pickle
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup

import tensorflow_cpu as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# CONFIG
# =========================
MAX_SEQ_LEN = 50
NUM_WORDS = 20000

# ABSOLUTE PATHS FOR HUGGINGFACE DOCKER
TFIDF_PATH = "/app/app/tfidf_vectorizer.pkl"
LOGREG_PATH = "/app/app/log_reg.pkl"
TOKENIZER_PATH = "/app/app/tokenizer.pkl"
CNN_MODEL_PATH = "/app/app/advanced_cnn_model.h5"

LABEL_MAP = {0: "Real", 1: "Fake"}

# =========================
# LOADING HELPERS (CACHED)
# =========================
@st.cache_resource
def load_tfidf_and_logreg():
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(LOGREG_PATH, "rb") as f:
        logreg = pickle.load(f)
    return tfidf, logreg


@st.cache_resource
def load_tokenizer_dict():
    with open(TOKENIZER_PATH, "rb") as f:
        tok_dict = pickle.load(f)
    return tok_dict


@st.cache_resource
def load_cnn_model():
    model = load_model(CNN_MODEL_PATH)
    return model


# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# TOKENIZER → SEQUENCES
# =========================
def text_to_sequence(text: str, tok_dict: dict, max_len: int = MAX_SEQ_LEN):
    word_index = tok_dict["word_index"]
    num_words = tok_dict.get("num_words", NUM_WORDS)

    cleaned = clean_text(text)
    tokens = cleaned.split()

    seq = []
    for w in tokens:
        idx = word_index.get(w)
        if idx is not None and idx < num_words:
            seq.append(idx)

    if len(seq) == 0:
        seq = [0]

    padded = pad_sequences(
        [seq],
        maxlen=max_len,
        padding="post",
        truncating="post"
    )
    return padded


# =========================
# PREDICT FUNCTIONS
# =========================
def decode_label(y_int: int) -> str:
    return LABEL_MAP.get(y_int, "Unknown")


def predict_with_logreg(text: str):
    tfidf, logreg = load_tfidf_and_logreg()
    cleaned = clean_text(text)

    X = tfidf.transform([cleaned])
    prob = logreg.predict_proba(X)[0, 1]

    label = int(prob >= 0.5)
    return decode_label(label), float(prob)


def predict_with_cnn(text: str):
    tok_dict = load_tokenizer_dict()
    model = load_cnn_model()

    seq = text_to_sequence(text, tok_dict)
    prob = model.predict(seq, verbose=0)[0, 0]

    label = int(prob >= 0.5)
    return decode_label(label), float(prob)


# =========================
# STREAMLIT PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

# THEME SELECTOR
theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        textarea, .stTextInput>div>div>input { background-color: #1b1f24 !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)

# INPUT MODE SELECTOR
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Input Mode")
input_mode = st.sidebar.radio("Provide news via:", ["Write text", "Paste URL"])

# TRENDING NEWS LINKS
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Trending News")
st.sidebar.markdown("""
🔗 [BBC News](https://www.bbc.com/news)  
🔗 [CNN](https://www.cnn.com)  
🔗 [Reuters](https://www.reuters.com/world/)  
🔗 [Fox News](https://www.foxnews.com)  
🔗 [NDTV](https://www.ndtv.com)  
""")

st.sidebar.markdown("---")
st.sidebar.caption("Created by Bhavya Pandya 🚀")


# =========================
# MAIN UI
# =========================
st.title("📰 Fake News Detection")
st.caption("Logistic Regression + CNN (20k vocab, seq_len=50)")

st.write("Paste a news article or URL and compare predictions:")

user_text = ""

# TEXT MODE
if input_mode == "Write text":
    user_text = st.text_area("✍️ Enter text:", height=200)

# URL MODE
else:
    url = st.text_input("🔗 Paste URL:")
    if url.strip():
        try:
            with st.spinner("Fetching article…"):
                html = requests.get(url, timeout=10).text
                soup = BeautifulSoup(html, "html.parser")
                paragraphs = soup.find_all("p")
                extracted = "\n".join([p.get_text() for p in paragraphs])
                user_text = extracted.strip()

            st.success("Article loaded!")
            st.text_area("Extracted text:", value=user_text[:2000], height=200)

        except Exception as e:
            st.error(f"Error fetching article: {e}")


# =========================
# BUTTONS
# =========================
col1, col2 = st.columns(2)
lr_result = None
cnn_result = None

with col1:
    if st.button("🔎 Predict with Logistic Regression"):
        if user_text.strip():
            with st.spinner("Running Logistic Regression..."):
                lr_result = predict_with_logreg(user_text)
        else:
            st.warning("Enter text first.")

with col2:
    if st.button("🧠 Predict with CNN"):
        if user_text.strip():
            with st.spinner("Running CNN..."):
                cnn_result = predict_with_cnn(user_text)
        else:
            st.warning("Enter text first.")


# =========================
# RESULTS
# =========================
st.markdown("---")

if lr_result:
    label, prob = lr_result
    st.subheader("📌 Logistic Regression Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability of Fake:** {prob:.3f}")

if cnn_result:
    label, prob = cnn_result
    st.subheader("🧠 CNN Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability of Fake:** {prob:.3f}")

st.markdown("---")
st.caption("*Educational tool. Verify news from reliable sources.*")
