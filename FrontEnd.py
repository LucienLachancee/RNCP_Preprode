import streamlit as st
from audiorecorder import audiorecorder
import tempfile
from BackEnd import generate_image_from_audio, audio_to_emotions
import io

st.set_page_config(page_title="🎙️ Générateur d'image à partir de l'audio", layout="centered")

st.title("🎧 Génère une image et des émotions depuis ta voix !")

st.markdown("### 1. Enregistre ton message 🎤")
audio = audiorecorder("📢 Appuie pour parler", "⏹️ Relâche pour arrêter")

if len(audio) > 0:
    st.success("✅ Audio capturé !")

    # Export audio dans un buffer mémoire
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    audio_bytes = buf.getvalue()

    # Lecture audio dans Streamlit
    st.audio(audio_bytes, format="audio/wav")

    # Sauvegarde audio temporaire sur disque
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

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
    st.info("🎙️ Appuie sur le bouton ci-dessus pour enregistrer un message.")
