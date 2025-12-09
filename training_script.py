# training_script.py
# Final training + export script for Fake News Detection models

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# 1. LOAD YOUR DATASET
# ----------------------------------------------------------
df = pd.read_csv("train.csv")  # change file name if needed
X_train_text = df["text"].astype(str)
y_train = df["label"]

# ----------------------------------------------------------
# 2. TRAIN TOKENIZER
# ----------------------------------------------------------
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

with open("app/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✔ tokenizer.pkl saved")

# ----------------------------------------------------------
# 3. TF-IDF + LOGISTIC REGRESSION
# ----------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(X_train_text)

X_vec = tfidf.transform(X_train_text)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_vec, y_train)

with open("app/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("app/log_reg.pkl", "wb") as f:
    pickle.dump(logreg, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✔ tfidf_vectorizer.pkl + log_reg.pkl saved")

# ----------------------------------------------------------
# 4. LOAD YOUR ORIGINAL CNN AND EXPORT LIGHT VERSION
# ----------------------------------------------------------
cnn = load_model("advanced_cnn_model.h5", compile=False)

# Save clean, CPU-compatible version (no optimizer state)
cnn.save("app/cnn_export.h5", include_optimizer=False)

print("✔ cnn_export.h5 saved (size reduced, deploy-ready)")
print("🎉 ALL MODELS EXPORTED SUCCESSFULLY")
