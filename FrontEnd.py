import streamlit as st
import tempfile
from BackEnd import generate_image_from_audio, audio_to_emotions

st.set_page_config(page_title="🎙️ Générateur d'image à partir de l'audio", layout="centered")

st.title("🎧 Génère une image et des émotions depuis ta voix !")

st.markdown("### 1. Dépose ton fichier audio 🎵 (formats pris en charge : wav, mp3...)")
uploaded_file = st.file_uploader("Glisser-déposer ou parcourir", type=["wav", "mp3", "ogg", "flac", "m4a"])

if uploaded_file is not None:
    st.success("✅ Fichier audio chargé !")

    # Sauvegarde du fichier uploadé temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    # Lecture audio
    st.audio(temp_path, format="audio/wav")

    # Traitement audio
    st.markdown("### 2. Résultat 📊🖼️")
    with st.spinner("⏳ Analyse audio en cours..."):
        transcription, prompt_generated, image_paths = generate_image_from_audio(temp_path)
        emotions = audio_to_emotions(transcription)

    # Affichage transcription
    st.subheader("📝 Transcription :")
    st.write(transcription)

    # Affichage émotions
    st.subheader("💡 Émotions détectées :")
    for emotion, value in emotions.items():
        st.write(f"- **{emotion.capitalize()}** : {round(value * 100, 2)}%")

    # Affichage images
    st.subheader("🖼️ Image(s) générée(s) :")
    for img_path in image_paths:
        st.image(img_path, use_container_width=True)

else:
    st.info("📁 Dépose un fichier audio (wav, mp3...) pour commencer.")
