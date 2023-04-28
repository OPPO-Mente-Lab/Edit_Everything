from PIL import Image
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline




img_url = "image/overture-creations-5sI6fQgYIuo.png"
mask_url = "image/overture-creations-5sI6fQgYIuo_mask.png"

init_image = Image.open(img_url).resize((512, 512))
mask_image = Image.open(mask_url).resize((512, 512))
print('loading stable diffusion')
diffusion_path='../StableDiffusion/stable-diffusion-inpainting'
pipe = StableDiffusionInpaintPipeline.from_pretrained(diffusion_path,torch_dtype=torch.float16)
pipe = pipe.to("cuda")
print(f'Info:Load StableDiffusionInpaintPipeline form {diffusion_path}')


if __name__=='__main__':
    prompt = "一只坐在那里的猫"
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]