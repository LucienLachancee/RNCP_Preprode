# backend.py
import time
import os
import requests

from dotenv import load_dotenv
import openai
from groq import Groq
from mistralai import Mistral

import base64

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")
api_key = os.environ["MISTRAL_API_KEY"]




def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()
    


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
    #prompt_text = "An ethereal character, lost in a dreamlike landscape of pastel colors, endlessly chewing, using a fork to taste a variety of dishes that seem to appear and disappear within a delicate scene, with surrealist painting accents and a dreamy atmosphere."
    for chunk in completion:
        if chunk.choices[0].delta.content:
            prompt_text += chunk.choices[0].delta.content

    # Agent d’image


    images = [] # Liste pour stocker les chemins des images générées

    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:  
            print(f"Error: {e}")
            return None
        

        
    base64_image = encode_image('user_pdp.jpg')




    

    print(f"Début de la génération d'image avec mistral ...")

    api_key = os.environ["MISTRAL_API_KEY"]

    # Specify model
    model = "pixtral-12b-2409"

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    # Define the messages for the chat
    messages = [
    {
    "role": "user",
    "content": [
    {
    "type": "text",
    "text": prompt_text
    },
    {
    "type": "image_url",
    "image_url": f"data:image/jpeg;base64,{base64_image}" 
    }
    ]
    }
    ]

    # Get the chat response
    chat_response = client.chat.complete(
    model=model,
    messages=messages
    )                       
    
    

    image_base64 = chat_response.choices[0].message.content.strip()

    # Nettoie un éventuel préfixe comme 'data:image/png;base64,'
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]

    try:
        image_bytes = base64.b64decode(image_base64)
        image_path = "generated_image_clipdrop_avec_pdp.png"

        with open(image_path, 'wb') as file:
            file.write(image_bytes)

        images.append(image_path)
        print(f"Image Clipdrop générée et sauvegardée sous {image_path}")

    except Exception as e:
        print(f"Erreur de décodage de l'image base64 : {e}")







    images.append(image_path)
    print(f"Image Clipdrop générée et sauvegardée sous {image_path}")
    

    total_end_time = time.time() # Fin totale de l'exécution de la fonction

    print(f"Temps de génération d'image avec ClipDrop : {total_end_time - start_time:.2f} secondes")

    return transcription.text, prompt_text, images

