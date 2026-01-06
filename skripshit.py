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
    /* Background utama */
    .stApp {
        background: linear-gradient(135deg, #fbd3e9, #fce7f3);
        font-family: "Segoe UI", sans-serif;
    }

    /* Container putih (card effect) */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 16px;
    }

    /* Judul */
    h1, h2, h3 {
        color: #9d174d;
    }

    /* Tombol */
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
        color: white;
    }

    /* Text area */
    textarea {
        border-radius: 12px !important;
        border: 1px solid #f9a8d4 !important;
    }

    /* Tabel */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Success & error box */
    .stAlert {
        border-radius: 12px;
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
    return pd.read_csv("youtube_data_fix.csv")

st.write("✅ App started")

# ===============================
# LOAD MODEL (AMAN UNTUK CLOUD)
# ===============================
try:
    model, tfidf = load_model()
    st.write("✅ Model loaded")
except Exception as e:
    st.error("❌ Model gagal dimuat di server")
    st.error(e)
    st.stop()

# ===============================
# LOAD DATA
# ===============================
try:
    df = load_data()
    st.write("✅ Data loaded, jumlah:", len(df))
except Exception as e:
    st.error("❌ Dataset gagal dimuat")
    st.error(e)
    st.stop()

# ===============================
# CLEAN LABEL
# ===============================
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].notna()]
df["label"] = df["label"].astype(int)
st.write("✅ Label cleaned")

# ===============================
# RINGKASAN DATASET
# ===============================
st.subheader("📌 Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Data", len(df))
col2.metric("Sentimen Negatif (0)", (df["label"] == 0).sum())
col3.metric("Sentimen Positif (1)", (df["label"] == 1).sum())


THRESHOLDS = [0.11, 0.30, 0.50, 0.70, 0.90]

# ===============================
# PREPROCESSING
# ===============================
abbreviations = {
    'yg': 'yang',
    'gak': 'tidak',
    'ga': 'tidak',
    'dlm': 'dalam',
    'bgt': 'banget',
    'tgl': 'tanggal',
    'utk': 'untuk',
    'sdh': 'sudah',
    'gakpapa': 'tidak apa-apa',
    'krn': 'karena',
    'cm' : 'cuman',
    'trus' : 'terus',
    'pa' : 'Pak',
    'drugikan' : 'dirugikan',
    'bubiar' : 'bubar'

}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def replace_abbreviations(text):
    words = text.split()
    return " ".join([abbreviations.get(w, w) for w in words])

# ===============================
# HEADER
# ===============================
st.title("📊 Analisis Sentimen Program Makanan Bergizi (MBG)")

st.write("""
Dashboard ini menampilkan hasil implementasi **Logistic Regression**
dalam menganalisis sentimen komentar YouTube terkait
**Program Makanan Bergizi (MBG)** dengan pendekatan
**SMOTE dan threshold tuning**.
""")

# ===============================
# DATASET BERDASARKAN LABEL
# ===============================
st.subheader("📂 Dataset Berdasarkan Label")

with st.expander("🔴 Lihat Data Sentimen Negatif (Label 0)"):
    st.dataframe(df[df["label"] == 0][["cleaned_comment", "label"]].head(10))

with st.expander("🟢 Lihat Data Sentimen Positif (Label 1)"):
    st.dataframe(df[df["label"] == 1][["cleaned_comment", "label"]].head(10))

# ===============================
# DISTRIBUSI DATA
# ===============================
st.subheader("📈 Distribusi Kelas Sentimen (Sebelum SMOTE)")
st.bar_chart(df["label"].value_counts())

st.caption("Dataset tidak seimbang sehingga digunakan metode SMOTE.")

# ===============================
# DISTRIBUSI DATA SETELAH SMOTE
# ===============================
st.subheader("📊 Distribusi Kelas Sentimen (Setelah SMOTE)")

max_count = df["label"].value_counts().max()

smote_dist = pd.DataFrame({
    "label": ["Negatif (0)", "Positif (1)"],
    "Jumlah Data": [max_count, max_count]
})

st.bar_chart(smote_dist.set_index("label"))

st.caption(
    "Distribusi kelas setelah penerapan SMOTE pada data latih, "
    "menunjukkan jumlah data yang seimbang antar kelas."
)


# ===============================
# PREDIKSI KOMENTAR
# ===============================
st.subheader("📝 Prediksi Sentimen Komentar MBG")

input_text = st.text_area(
    "Masukkan komentar terkait Program Makanan Bergizi:",
    placeholder="Contoh: Program ini sangat membantu anak-anak sekolah..."
)

if st.button("🔍 Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        # preprocessing
        text_clean = replace_abbreviations(clean_text(input_text))
        X = tfidf.transform([text_clean])
        prob = model.predict_proba(X)[0][1]

        st.write(f"**Probabilitas Sentimen Positif:** {prob:.4f}")

        # ===============================
        # KESIMPULAN AKHIR (THRESHOLD TERBAIK)
        # ===============================
        final_label = "Positif" if prob >= 0.30 else "Negatif"

        st.subheader("📌 Kesimpulan Akhir")
        if final_label == "Positif":
            st.success("✅ Komentar ini diklasifikasikan sebagai **SENTIMEN POSITIF** berdasarkan threshold 0.30")
        else:
            st.error("❌ Komentar ini diklasifikasikan sebagai **SENTIMEN NEGATIF** berdasarkan threshold 0.30")

        # ===============================
        # PERBANDINGAN THRESHOLD
        # ===============================
        results = []
        thresholds = [0.11, 0.30, 0.50, 0.70, 0.90]

        for t in thresholds:
            label = "Positif" if prob >= t else "Negatif"
            results.append({
                "Threshold": t,
                "Hasil Klasifikasi": label
            })

        result_df = pd.DataFrame(results)

        st.subheader("📊 Perbandingan Hasil Berdasarkan Threshold")
        st.table(result_df)


# ===============================
# INFORMASI MODEL
# ===============================
st.subheader("ℹ️ Informasi Model")

st.write("""
- **Algoritma**: Logistic Regression  
- **Penanganan Data Tidak Seimbang**: SMOTE  
- **Threshold yang Diuji**: 0.11, 0.30, 0.50, 0.70, 0.90  
- **Threshold Optimal**: 0.30 (F1-Score tertinggi)  
""")

st.info("""
Keterangan Label:
- 0 = Sentimen Negatif
- 1 = Sentimen Positif
""")
st.warning("""
Keterbatasan Model:
- Analisis hanya mencakup dua kelas sentimen (positif dan negatif)
- Tidak mempertimbangkan konteks sarkasme atau ironi
""")
st.write("✅ App started")

model, tfidf = load_model()
st.write("✅ Model loaded")

df = load_data()
st.write("✅ Data loaded, jumlah:", len(df))

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].notna()]
df["label"] = df["label"].astype(int)
st.write("✅ Label cleaned")

