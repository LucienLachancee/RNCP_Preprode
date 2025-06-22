import requests
import base64
import os
from dotenv import load_dotenv
import json

# Charger les variables d'environnement (votre API KEY)
load_dotenv()

def image_file_to_base64(image_path):
    """Encode une image en chaîne de caractères base64."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

# --- CONFIGURATION ---
# ✅ On utilise l'endpoint IMAGE-TO-IMAGE
url = "https://api.segmind.com/v1/kandinsky2.2-img2img" 
api_key = os.environ.get("KANDINSKY_API_KEY")

if not api_key:
    raise ValueError("Clé API non trouvée. Assurez-vous qu'elle est dans votre fichier .env sous le nom KANDINSKY_API_KEY")

# --- PRÉPARATION DES DONNÉES ---
data = {
    # Le prompt qui va guider la TRANSFORMATION de l'image
    "prompt": "this guy is a wizard like gandalf in lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
    "negative_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands",
    
    # ✅ Le paramètre pour l'image de départ s'appelle "image"
    "image": image_file_to_base64('user_pdp.jpg'), 
    
    # Paramètres du modèle
    "samples": 1,
    "num_inference_steps": 50,
    "strength": 0.75, # Force de la transformation (0 = image originale, 1 = transformation complète)
    "guidance_scale": 7.5,
    "seed": 9863172
}

headers = {'x-api-key': api_key}

# --- APPEL API ET TRAITEMENT DE LA RÉPONSE ---
print("🚀 Envoi de la requête à l'API Segmind (img2img)...")
response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    # L'API img2img de Segmind renvoie directement l'image binaire
    # Pas besoin de parser un JSON
    image_bytes = response.content 
    image_path = "kandinsky_image_transformee.jpeg" # On garde le format jpeg

    with open(image_path, 'wb') as file:
        file.write(image_bytes)

    print(f"✅ Image transformée enregistrée sous : {image_path}")

else:
    # En cas d'erreur, on affiche le statut et le message d'erreur
    print(f"❌ Erreur lors de l'appel API : {response.status_code}")
    print(f"Message : {response.text}")