import streamlit as st
import joblib
import numpy as np

# === LOAD MODEL & VECTORIZER ===
MODEL_PATH = "model_svm.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except:
    st.error("❌ File model atau vectorizer tidak ditemukan. Pastikan sudah di-upload ke folder yang sama dengan main.py.")
    st.stop()

# === UI STREAMLIT ===
st.set_page_config(page_title="Sentiment Review E-Commerce", layout="centered")

st.title("🛍️ Analisis Sentimen Review E-Commerce")
st.write("Masukkan review produk untuk memprediksi sentimen (positif / negatif).")

review_input = st.text_area("Tulis review di sini:", height=150)

if st.button("Prediksi"):
    if review_input.strip() == "":
        st.warning("Tolong masukkan teks review terlebih dahulu.")
    else:
        data = vectorizer.transform([review_input])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.success("💚 Sentimen: **POSITIF**")
        else:
            st.error("❤️ Sentimen: **NEGATIF**")
