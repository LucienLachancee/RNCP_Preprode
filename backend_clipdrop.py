# backend.py
import time
import os
import requests

from dotenv import load_dotenv
import openai
from groq import Groq


import base64

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")




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


        
    try:
        # print(f"Début de la génération d'image avec Clipdrop ...")

        # clipdrop_response = requests.post('https://clipdrop-api.co/text-to-image/v1',
        # files = {
        #     'prompt': (None, prompt_text, 'text/plain')
        # },
        # headers = { 'x-api-key': CLIPDROP_API_KEY}
        # )
        # if (clipdrop_response.ok):
        #     image_bytes_response = clipdrop_response.content 

        #     # # Télécharger l'image depuis l'URL et la sauvegarder localement
        #     # image_bytes_response = requests.get(image_url, stream=True)
        #     # image_bytes_response.raise_for_status() # Lève une exception si le téléchargement échoue

        #     image_path = "generated_image_clipdrop.png"


        #     if clipdrop_response.ok: # Equivaut à clipdrop_response.status_code == 200
            
        #         with open(image_path, 'wb') as file:
        #             file.write(clipdrop_response.content)
        #         images.append(image_path)
        #         print(f"Image Clipdrop générée et sauvegardée sous {image_path}")
        #     else:
        #         # Cette branche est normalement atteinte si raise_for_status() n'a pas levé d'exception
        #         print(f"Erreur lors de la génération d'image Clipdrop (status {clipdrop_response.status_code}): {clipdrop_response.text}")
        #         raise Exception(f"Erreur lors de la génération d'image Clipdrop: {clipdrop_response.text}")
            
################################### Clip drop sketch image #####################################################################

        print(f"Début de la génération d'image avec Clipdrop ...")
        
        clipdrop_response = requests.post('https://clipdrop-api.co/sketch-to-image/v1/sketch-to-image',
        files = {
                    'sketch_file': ('user_pdp.jpg', open('user_pdp.jpg', 'rb'), 'image/jpg'),
                    "prompt": (None, prompt_text)
                        },
        headers = { 'x-api-key': CLIPDROP_API_KEY}
        )
        if (clipdrop_response.ok):
            image_bytes_response = clipdrop_response.content 

            # # Télécharger l'image depuis l'URL et la sauvegarder localement
            # image_bytes_response = requests.get(image_url, stream=True)
            # image_bytes_response.raise_for_status() # Lève une exception si le téléchargement échoue

            image_path = "generated_image_clipdrop_avec_pdp.png"


            if clipdrop_response.ok: # Equivaut à clipdrop_response.status_code == 200
            
                with open(image_path, 'wb') as file:
                    file.write(clipdrop_response.content)
                images.append(image_path)
                print(f"Image Clipdrop générée et sauvegardée sous {image_path}")
            else:
                # Cette branche est normalement atteinte si raise_for_status() n'a pas levé d'exception
                print(f"Erreur lors de la génération d'image Clipdrop (status {clipdrop_response.status_code}): {clipdrop_response.text}")
                raise Exception(f"Erreur lors de la génération d'image Clipdrop: {clipdrop_response.text}")

    
        else:
            clipdrop_response.raise_for_status()


    except requests.exceptions.RequestException as e:
        print(f"Erreur réseau ou HTTP lors de la génération d'image Clipdrop: {e}")
        raise # Re-lancer l'exception pour que le programme s'arrête ou soit géré en amont
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de la génération d'image Clipdrop: {e}")
        raise # Re-lancer l'exception



    total_end_time = time.time() # Fin totale de l'exécution de la fonction

    print(f"Temps de génération d'image avec ClipDrop : {total_end_time - start_time:.2f} secondes")

    return transcription.text, prompt_text, images, image_bytes_response

