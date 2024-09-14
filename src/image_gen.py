import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline, StableDiffusionPipeline

# Current options for text-to-image models
text_2_img_models = [
    "black-forest-labs/FLUX.1-schnell",
    "lambdalabs/miniSD-diffusers",  # https://huggingface.co/lambdalabs/miniSD-diffusers
    "OFA-Sys/small-stable-diffusion-v0",  # https://huggingface.co/OFA-Sys/small-stable-diffusion-v0
    "stabilityai/sdxl-turbo",  # https://huggingface.co/stabilityai/sdxl-turbo
    "stabilityai/stable-diffusion-3-medium",  # https://huggingface.co/stabilityai/stable-diffusion-3-medium
]


def get_flux_pipe():
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


def get_small_sd_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        torch_dtype=torch.float16,
        cache_dir="models",
    )
    return pipe


def get_mini_sd_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "lambdalabs/miniSD-diffusers", cache_dir="models"
    )
    pipe = pipe.to("cuda")

    return pipe


def get_sd_turbo_pipe():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir="models",
    )
    pipe = pipe.to("cuda")
    return pipe


def generate_img(
    model_name: str = "sdxl-turbo",
    prompt: str = "A flashy bottle that stands out from the rest.",
):
    if model_name == "sdxl-turbo":
        pipe = get_sd_turbo_pipe()
    else:
        pipe = get_mini_sd_pipe()

    # pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    # pipe.to("cuda")
    # Play with these parameters to get different results
    image = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        height=1024,
        width=1024,
        num_inference_steps=4,
        max_sequence_length=256,
    ).images[0]
    image.save(f"text2img_tests/{model_name}.jpg")


generate_img()
