# backend.py
import time
import os
# from dotenv import load_dotenv
from groq import Groq
from mistralai.models import ToolFileChunk
from mistralai import Mistral
import json
import math
import streamlit as st



# load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

#pour streamlit cloud
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
mistral_client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])




def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()
    
def softmax(prediction):
    output = {}
    for sentiment,predicted_value in prediction.items():
        output[sentiment] = math.exp(predicted_value*10) / sum([math.exp(value*10) for value in prediction.values()])
    return output


def generate_image_from_audio(file_path):

    start_time = time.time() 
    # Transcription audio
    with open(file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt="Specify context or spelling",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="fr",
            temperature=0.0
        )

    # Génération du prompt texte
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",
             "content": read_file("./context.txt")},

            {"role": "user", 
              "content": transcription.text}
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True
    )

    prompt_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            prompt_text += chunk.choices[0].delta.content

    # Agent d’image
    image_agent = mistral_client.beta.agents.create(
        model="mistral-medium-2505",
        name="Image Generation Agent",
        description="Agent used to generate images.",
        instructions="Use the image generation tool when you have to create images.",
        tools=[{"type": "image_generation"}],
        completion_args={
            "temperature": 1,
            "top_p": 0.95
        }
    )

    response = mistral_client.beta.conversations.start(
        agent_id=image_agent.id,
        inputs=prompt_text
    )

    images = []
    for i, chunk in enumerate(response.outputs[-1].content):
        if isinstance(chunk, ToolFileChunk):
            file_bytes = mistral_client.files.download(file_id=chunk.file_id).read()
            image_path = f"image_generated_{i}.png"
            with open(image_path, "wb") as f:
                f.write(file_bytes)
            images.append(image_path)

    end_time = time.time()

    print(f"Temps total d'exécution avec {image_agent.model}: {end_time - start_time:.2f} secondes")

    return transcription.text, prompt_text, images

def audio_to_emotions (transcription_text):
    model = "mistral-large-latest"

    client = mistral_client

    chat_response = client.chat.complete(
        model = model,
        messages = [
            {
                "role": "system",
                "content": read_file(file_path="context_emotion.txt"),
            },
            {
                "role": "user",
                "content": f"Analyse le texte ci-dessous. Ta réponse doit être un dictionnaire JSON valide avec des émotions en clé et des scores entre 0 et 1 en valeur. Ne mets pas de texte, uniquement du JSON : {transcription_text}",
            },
        ],
        response_format={"type": "json_object",}
    )
    prediction = chat_response.choices[0].message.content
    prediction_dict = json.loads(prediction)
  
    return softmax(prediction_dict)