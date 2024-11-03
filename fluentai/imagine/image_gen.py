import hashlib

import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline, StableDiffusionPipeline

# Current options for text-to-image models
text_2_img_models = [
    "stabilityai/stable-diffusion-3.5-medium" "black-forest-labs/FLUX.1-schnell",
    "lambdalabs/miniSD-diffusers",  # https://huggingface.co/lambdalabs/miniSD-diffusers
    "OFA-Sys/small-stable-diffusion-v0",  # https://huggingface.co/OFA-Sys/small-stable-diffusion-v0
    "stabilityai/sdxl-turbo",  # https://huggingface.co/stabilityai/sdxl-turbo
    "stabilityai/stable-diffusion-3-medium",  # https://huggingface.co/stabilityai/stable-diffusion-3-medium
]


def _get_flux_pipe():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,  # torch.float16,
        # variant="fp16",
        cache_dir="models",
    )
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.to(torch.float16)

    return pipe


def _get_small_sd_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        torch_dtype=torch.float16,
        cache_dir="models",
    )
    return pipe


def _get_mini_sd_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "lambdalabs/miniSD-diffusers", cache_dir="models"
    )
    pipe = pipe.to("cuda")

    return pipe


def _get_sd_turbo_pipe():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir="models",
    )
    pipe = pipe.to("cuda")
    return pipe


def _get_sd_medium_pipe():
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
        cache_dir="models",
    )
    pipe = pipe.to("cuda")

    return pipe

    image = pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    image.save("capybara.png")


def generate_short_code_sha256(prompt: str, length: int = 8) -> str:
    """
    Generates a short code for a given prompt using SHA256 hashing.

    Args:
        prompt (str): The input sentence or prompt.
        length (int): Desired length of the short code. Default is 8.

    Returns
    -------
        str: A short hexadecimal code representing the prompt.
    """
    # Create a SHA256 hash object
    hash_object = hashlib.sha256(prompt.encode("utf-8"))

    # Get the hexadecimal digest of the hash
    hex_digest = hash_object.hexdigest()

    # Truncate the hash to the desired length
    short_code = hex_digest[:length]

    return short_code


def generate_img(
    model_name: str = "sdxl-turbo",
    prompt: str = "A flashy bottle that stands out from the rest.",
):
    """
    Generate an image from a given prompt using a text-to-image model.

    Parameters
    ----------
    model_name : str, optional
        _description_, by default "sdxl-turbo"
    prompt : str, optional
        _description_, by default "A flashy bottle that stands out from the rest."
    """
    pipe = _get_sd_medium_pipe()

    # pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    # pipe.to("cuda")
    # Play with these parameters to get different results
    image = pipe(
        prompt=prompt,
        # guidance_scale=0.0,
        # height=1024,
        # width=1024,
        num_inference_steps=4.5,
        # max_sequence_length=256,
    ).images[0]

    # Temporary solution to save the prompt in the image filename
    prompt_code = generate_short_code_sha256(prompt)

    image.save(f"img/text2img_tests/{model_name}_{prompt_code}.jpg")


generate_img()
