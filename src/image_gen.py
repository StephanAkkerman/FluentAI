import torch
from diffusers import StableDiffusionPipeline


# https://huggingface.co/OFA-Sys/small-stable-diffusion-v0
def small_stable():
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        torch_dtype=torch.float16,
        cache_dir="models",
    )
    pipe = pipe.to("cuda")

    prompt = "A flashy bottle that stands out from the rest."
    image = pipe(prompt).images[0]

    image.save("apple.png")


def mini_sd():
    pipe = StableDiffusionPipeline.from_pretrained(
        "lambdalabs/miniSD-diffusers", cache_dir="models"
    )
    pipe = pipe.to("cuda")

    prompt = "A flashy bottle that stands out from the rest."
    image = pipe(prompt, width=256, height=256).images[0]
    image.save("test.jpg")


mini_sd()
