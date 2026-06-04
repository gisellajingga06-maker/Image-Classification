import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.set_page_config(page_title="Klasifikasi Hewan", page_icon="🐾")

st.title("🐾 Klasifikasi Gambar Hewan")
st.write("Upload gambar untuk memprediksi apakah gambar termasuk kelinci atau lumba-lumba.")

MODEL_PATH = "model_hewan.keras"
class_names = ["kelinci", "lumba2"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    img_resized = img.resize((227, 227))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    hasil = model.predict(img_array, verbose=0)
    prediksi = np.argmax(hasil)

    st.subheader("Hasil Prediksi")
    st.success(f"{class_names[prediksi]}")

    st.subheader("Probabilitas")
    st.write(hasil)
