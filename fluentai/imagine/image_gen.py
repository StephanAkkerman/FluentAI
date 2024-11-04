import hashlib

# from fluentai.utils.logger import logger
import logging

import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline, StableDiffusionPipeline

logger = logging.getLogger(__name__)

# Current options for text-to-image models
text_2_img_models = [
    "stabilityai/stable-diffusion-3.5-medium",
    "black-forest-labs/FLUX.1-schnell",
    "lambdalabs/miniSD-diffusers",  # https://huggingface.co/lambdalabs/miniSD-diffusers
    "OFA-Sys/small-stable-diffusion-v0",  # https://huggingface.co/OFA-Sys/small-stable-diffusion-v0
    "stabilityai/sdxl-turbo",  # https://huggingface.co/stabilityai/sdxl-turbo
    "stabilityai/stable-diffusion-3-medium-diffusers",  # https://huggingface.co/stabilityai/stable-diffusion-3-medium
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

    # https://github.com/Stability-AI/sd3.5/blob/main/requirements.txt

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
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


def test():
    import torch
    from diffusers import (
        BitsAndBytesConfig,
        SD3Transformer2DModel,
        StableDiffusion3Pipeline,
    )

    model_id = "stabilityai/stable-diffusion-3.5-medium"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        cache_dir="models",
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, transformer=model_nf4, torch_dtype=torch.bfloat16, cache_dir="models"
    )
    print(pipeline.device)
    pipeline.to("cuda")
    print(pipeline.device)

    prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

    image = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
        max_sequence_length=512,
    ).images[0]
    image.save("whimsical.png")


def generate_img(
    model_name: str = "sdxl-turbo",
    prompt: str = "A flashy bottle that stands out from the other bottles.",
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
    pipe = _get_sd_turbo_pipe()

    # pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    # pipe.to("cuda")
    # Play with these parameters to get different results
    logger.info(f"Generating image for prompt: {prompt}")
    image = pipe(
        prompt=prompt,
        # guidance_scale=0.0,
        height=512,
        width=512,
        num_inference_steps=5,
        guidance_scale=4.5,
        # max_sequence_length=256,
    ).images[0]

    # Temporary solution to save the prompt in the image filename
    # prompt_code = generate_short_code_sha256(prompt)
    prompt_code = "test"

    image.save(f"img/text2img_tests/{model_name}_{prompt_code}.jpg")
    logger.info("Generated image!")


generate_img()
