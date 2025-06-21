import streamlit as st
import tempfile
from pydub import AudioSegment
from BackEnd import generate_image_from_audio, audio_to_emotions

st.set_page_config(page_title="🎙️ Générateur d'image à partir de l'audio", layout="centered")

st.title("🎧 Génère une image et des émotions depuis ta voix !")

st.markdown("### 1. Dépose ton fichier audio 🎵 (formats pris en charge : mp3, wav, ogg, m4a...)")
uploaded_file = st.file_uploader("Glisser-déposer ou parcourir", type=["mp3", "wav", "ogg", "m4a", "flac"])

if uploaded_file is not None:
    st.success("✅ Fichier audio chargé !")

    # Sauvegarde du fichier uploadé temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    # Conversion vers WAV avec pydub
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
        sound = AudioSegment.from_file(input_path)
        sound.export(temp_output.name, format="wav")
        wav_path = temp_output.name

    # Lecture audio dans Streamlit
    st.audio(wav_path, format="audio/wav")

    # Traitement audio
    st.markdown("### 2. Résultat 📊🖼️")
    with st.spinner("⏳ Analyse audio en cours..."):
        transcription, prompt_generated, image_paths = generate_image_from_audio(wav_path)
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
    st.info("📁 Dépose un fichier audio (mp3, wav, ogg, m4a...) pour commencer.")
