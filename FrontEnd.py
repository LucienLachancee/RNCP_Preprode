# app.py

import streamlit as st
import tempfile
#from BackEnd import generate_image_from_audio
from io import BytesIO
from PIL import Image

from backend_mistral_pixtral import generate_image_from_audio


st.title("🎙️➡️🖼️ Générateur d’image à partir d’un fichier audio")

uploaded_file = st.file_uploader("Téléversez un fichier audio (.m4a)", type=["m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    with st.spinner("Traitement en cours..."):
        transcription_text, prompt_generated, image_paths = generate_image_from_audio(temp_path)

    st.success("✅ Image générée avec succès !")

    st.markdown("### ✍️ Transcription")
    st.write(transcription_text)

    st.markdown("### 🖼️ Image générée")

    


    for path in image_paths:
        st.image(path, caption=path)


