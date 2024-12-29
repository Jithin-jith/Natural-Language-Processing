import os
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
import torch
load_dotenv()

# Replace 'your_access_token' with your Hugging Face access token
HF_API_key = os.getenv('HF_API_Key')

# Load the pre-trained model from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0", use_auth_token=HF_API_key)
pipe.to("cuda")  # Use GPU if available

# Define the text prompt
prompt = """A elephant"""

# Generate the image from the text prompt
image = pipe(prompt).images[0]

# Save the generated image
image.save("generated_image.png")

# Optionally display the image
image.show()