import torch
from diffusers import AutoPipelineForText2Image


def sd_turbo(prompt: str = "A flashy bottle that stands out from the rest."):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir="models",
    )
    pipe.to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save("test.jpg")


sd_turbo()
