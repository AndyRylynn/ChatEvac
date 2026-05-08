from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import os

_here = os.path.dirname(os.path.abspath(__file__))

controlnet = ControlNetModel.from_pretrained(os.path.join(_here, "checkpoint"), torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

pipeline.safety_checker = None

input_path = os.path.join(_here, "Material", "Input.png")
control_image = load_image(input_path)
prompt = "Identify building walls in the image; annotate pixels corresponding to walls and outdoor areas as white, annotate indoor areas as black, ignore indoor doors, and represent doors leading outdoors with red lines."

generator = torch.manual_seed(1225)
image = pipeline(prompt, num_inference_steps=30, generator=generator, image=control_image, controlnet_conditioning_scale=1.0).images[0]

output_path = os.path.join(_here, "Material", "Output.png")
image.save(output_path)
print("Generated image saved at:", output_path)
