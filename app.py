import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time
import pandas as pd
import plotly.graph_objects as go

# =========================
# CONFIG HALAMAN
# =========================
st.set_page_config(
    page_title="Pneumonia Detection | Naufal Ardra Anabil",
    page_icon="🫁",
    layout="wide"
)

# =========================
# CSS CUSTOM
# =========================
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 10px; }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        color: #6c757d;
        background: white;
        border-top: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# PATCH MODEL (SAFE LOAD)
# =========================
from tensorflow.keras.layers import Dense as KerasDense

class Dense(KerasDense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    file_name = "model_resnet50_pneumonia_terbaik.h5"
    path = os.path.join(os.path.dirname(__file__), file_name)

    if not os.path.exists(path):
        return None, "Model tidak ditemukan"

    try:
        with tf.keras.utils.custom_object_scope({
            "Dense": Dense,
            "quantization_config": None
        }):
            model = tf.keras.models.load_model(path, compile=False)

        return model, None

    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Logo_Gunadarma.png/600px-Logo_Gunadarma.png",
        width=150
    )

    st.title("🗂️ Informasi Peneliti")

    st.markdown("""
    **Nama:** Naufal Ardra Anabil  
    **NPM:** 51422215  
    **Prodi:** Informatika  
    **Universitas:** Gunadarma  
    """)

    st.divider()
    st.subheader("⚙️ Sistem")

    st.write("Model: ResNet-50")
    st.write("Framework: TensorFlow + Streamlit")

    if model:
        st.success("Model Loaded ✅")
    else:
        st.error(f"Model Error: {model_error}")

# =========================
# TITLE
# =========================
st.title("🫁 Pneumonia Detection System")
st.markdown("Klasifikasi Citra X-Ray Menggunakan Deep Learning (ResNet-50)")

# =========================
# TAB MENU
# =========================
tab1, tab2, tab3 = st.tabs(["🔍 Diagnosa", "📊 Performa", "📖 Panduan"])

# =========================
# TAB 1 - PREDIKSI
# =========================
with tab1:
    if model is None:
        st.error("Model tidak bisa dijalankan.")
    else:
        uploaded = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

        if uploaded:
            col1, col2 = st.columns(2)

            image = Image.open(uploaded)

            with col1:
                st.subheader("Input Gambar")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Hasil Analisis AI")

                with st.spinner("Menganalisis..."):

                    # preprocessing
                    img = image.convert("RGB").resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    # inference
                    start = time.time()
                    pred = model.predict(img_array)
                    end = time.time()

                    labels = ["Normal", "Pneumonia"]
                    idx = np.argmax(pred)
                    result = labels[idx]
                    confidence = float(np.max(pred)) * 100

                    # result output
                    if result == "Pneumonia":
                        st.error(f"HASIL: {result}")
                        st.warning("Segera konsultasi ke dokter.")
                    else:
                        st.success(f"HASIL: {result}")
                        st.info("Paru-paru terlihat normal.")

                    st.metric("Confidence", f"{confidence:.2f}%")
                    st.metric("Inference Time", f"{end - start:.4f} sec")

                    # chart plotly
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=labels,
                        y=pred[0] * 100
                    ))
                    fig.update_layout(title="Probabilitas Prediksi")
                    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2 - PERFORMANCE
# =========================
with tab2:
    st.header("📊 Evaluasi Model")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "92.4%")
    col2.metric("Dataset", "5,856 images")
    col3.metric("Split", "60/20/20")

    st.subheader("Pipeline CRISP-DM")
    st.write("""
    1. Business Understanding  
    2. Data Understanding  
    3. Data Preparation  
    4. Modeling (ResNet-50)  
    5. Evaluation  
    6. Deployment (Streamlit)
    """)

# =========================
# TAB 3 - GUIDE
# =========================
with tab3:
    st.header("📖 Cara Penggunaan")

    st.markdown("""
    - Upload gambar X-Ray dada  
    - Tunggu proses AI  
    - Lihat hasil prediksi  
    - Gunakan gambar yang jelas  
    """)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
Copyright © 2026 | Naufal Ardra Anabil | Skripsi Informatika
</div>
""", unsafe_allow_html=True)