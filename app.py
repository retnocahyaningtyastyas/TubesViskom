import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model sesuai pilihan optimizer
@st.cache_resource
def load_trained_model(optimizer_name):
    if optimizer_name == 'Adam':
        return load_model('adam_model.h5')
    elif optimizer_name == 'Nadam':
        return load_model('nadam_model.h5')
    elif optimizer_name == 'RMSProp':
        return load_model('rmsprop_model.h5')
    else:
        raise ValueError("Optimizer tidak dikenal.")

st.title("Aksara Jawa Classifier")

# Pilihan optimizer
optimizer_selected = st.selectbox("Pilih Optimizer", ["Adam", "Nadam", "RMSProp"])
model = load_trained_model(optimizer_selected)

# Upload gambar aksara
uploaded_file = st.file_uploader("Upload gambar aksara jawa", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # Preprocessing
    img = img.resize((224,224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']  # Sesuaikan dengan kelas kamu
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

   # Tampilkan hasil prediksi
    st.write(f"**Prediksi Aksara:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")