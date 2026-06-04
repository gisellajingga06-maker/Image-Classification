import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile

st.title("Klasifikasi CNN")

model_file = st.file_uploader(
    "Upload Model CNN (.keras)",
    type=["keras"]
)

image_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

# Ganti sesuai jumlah kelas model Anda
class_names = [
    "Kelas 1",
    "Kelas 2",
    "Kelas 3"
]

if model_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    model = tf.keras.models.load_model(model_path)

    st.success("Model berhasil dimuat")

    if image_file is not None:

        img = Image.open(image_file).convert("RGB")

        st.image(img, caption="Gambar Input", use_container_width=True)

        # Ukuran input model
        img_resize = img.resize((224, 224))

        # Konversi ke array
        img_array = np.array(img_resize)

        # Normalisasi
        img_array = img_array / 255.0

        # Tambah dimensi batch
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader("Hasil Klasifikasi")
        st.write(
            f"Prediksi: **{class_names[predicted_class]}**"
        )
        st.write(
            f"Tingkat Keyakinan: **{confidence:.2f}%**"
        )

        st.subheader("Probabilitas Tiap Kelas")

        for i, label in enumerate(class_names):
            st.write(
                f"{label}: {prediction[0][i]*100:.2f}%"
            )
