# import requests
# import pickle
# from dotenv import load_dotenv
# import os
# import base64



# load_dotenv()

# response = requests.post('https://clipdrop-api.co/text-to-image/v1',
#   files = {
#       'prompt': (None, 'shot of vaporwave fashion dog in miami', 'text/plain')
#   },
#   headers = { 'x-api-key': os.environ["CLIPDROP_API_KEY"]}
# )
# with open('response.pkl', 'wb') as f:  # open a text file
#     pickle.dump(response, f) # serialize the list


# with open("response.pkl", "rb") as file:
#     response = pickle.load(file)



# if (response.ok):
#   with open("image_clipdrop.png", "wb") as file:
#     file.write(response.content)

# else:
#   response.raise_for_status()



# img_encoded = base64.b64encode(response.content)
# print(img_encoded)
####################################################@##########################@


import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
#pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#pipeline.enable_xformers_memory_efficient_attention()



init_image = load_image("user_pdp.jpg")


prompt = "this guy is wizard like gandalf in lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]

image.save("kandinsky_output.png")
from PIL import Image

# Sauvegarder l'image générée
output_path = "kandinsky_output.png"
image.save(output_path)

print(f"L'image a été enregistrée ici : {output_path}")

make_image_grid([init_image, image], rows=1, cols=2)