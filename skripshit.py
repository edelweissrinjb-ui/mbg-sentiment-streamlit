import streamlit as st
import pandas as pd
import joblib
import re

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen MBG",
    page_icon="📊",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fbd3e9, #fce7f3);
        font-family: "Segoe UI", sans-serif;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 16px;
    }
    h1, h2, h3 {
        color: #9d174d;
    }
    div.stButton > button {
        background-color: #ec4899;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #db2777;
    }
    textarea {
        border-radius: 12px !important;
        border: 1px solid #f9a8d4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# LOAD MODEL & DATA
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("logreg_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

@st.cache_data
def load_data():
    return pd.read_csv("datafinal_youtube_data.csv")

# ===============================
# INIT
# ===============================
try:
    model, tfidf = load_model()
except Exception as e:
    st.error("❌ Model gagal dimuat")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error("❌ Dataset gagal dimuat")
    st.stop()

# ===============================
# CLEAN LABEL
# ===============================
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].notna()]
df["label"] = df["label"].astype(int)

# ===============================
# PREPROCESSING
# ===============================
abbreviations = {
    'yg': 'yang',
    'gak': 'tidak',
    'ga': 'tidak',
    'dlm': 'dalam',
    'bgt': 'banget',
    'utk': 'untuk',
    'sdh': 'sudah',
    'krn': 'karena',
    'trus': 'terus'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def replace_abbreviations(text):
    return " ".join([abbreviations.get(w, w) for w in text.split()])

# ===============================
# HEADER
# ===============================
st.title("📊 Analisis Sentimen Program Makanan Bergizi (MBG)")

st.write("""
Aplikasi ini menampilkan hasil implementasi **Logistic Regression**
untuk menganalisis sentimen komentar YouTube terkait
**Program Makanan Bergizi (MBG)** menggunakan pendekatan
**threshold tuning** pada model yang telah dilatih.
""")

# ===============================
# DATASET CONTOH
# ===============================
st.subheader("📂 Contoh Dataset Berdasarkan Label")

with st.expander("🔴 Contoh Sentimen Negatif"):
    st.dataframe(
        df[df["label"] == 0][["clean_text_expanded", "label"]].head(10)
    )

with st.expander("🟢 Contoh Sentimen Positif"):
    st.dataframe(
        df[df["label"] == 1][["clean_text_expanded", "label"]].head(10)
    )

# ===============================
# PENJELASAN SMOTE (TANPA VISUAL)
# ===============================
st.subheader("📌 Penanganan Data Tidak Seimbang")

st.write("""
Pada tahap **pelatihan model (offline)**, dataset yang tidak seimbang
ditangani menggunakan metode **SMOTE** untuk meningkatkan kemampuan
model dalam mengenali kelas minoritas.

Pada aplikasi ini, pengguna hanya ditampilkan **hasil prediksi akhir**
tanpa visualisasi proses penyeimbangan data.
""")

# ===============================
# PREDIKSI KOMENTAR
# ===============================
st.subheader("📝 Prediksi Sentimen Komentar")

input_text = st.text_area(
    "Masukkan komentar terkait Program Makanan Bergizi:",
    placeholder="Contoh: Program ini sangat membantu anak-anak sekolah"
)

if st.button("🔍 Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        text_cleaned = replace_abbreviations(clean_text(input_text))
        X = tfidf.transform([text_cleaned])
        prob = model.predict_proba(X)[0][1]

        st.write(f"**Probabilitas Sentimen Positif:** {prob:.4f}")

        final_label = "Positif" if prob >= 0.30 else "Negatif"

        st.subheader("📌 Hasil Klasifikasi")
        if final_label == "Positif":
            st.success("✅ Komentar diklasifikasikan sebagai **SENTIMEN POSITIF**")
        else:
            st.error("❌ Komentar diklasifikasikan sebagai **SENTIMEN NEGATIF**")

        thresholds = [0.11, 0.30, 0.50]
        results = [{
            "Threshold": t,
            "Hasil": "Positif" if prob >= t else "Negatif"
        } for t in thresholds]

        st.subheader("📊 Perbandingan Threshold")
        st.table(pd.DataFrame(results))

# ===============================
# INFORMASI MODEL
# ===============================
st.subheader("ℹ️ Informasi Model")

st.write("""
- **Algoritma**: Logistic Regression  
- **Pendekatan**: TF-IDF & Threshold Tuning  
- **Threshold Optimal**: 0.30  
""")

st.info("""
Keterangan Label:
- 0 = Sentimen Negatif
- 1 = Sentimen Positif
""")

st.warning("""
Keterbatasan:
- Hanya dua kelas sentimen
- Belum menangani sarkasme
""")
