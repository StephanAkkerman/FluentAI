import torch
from diffusers import DiffusionPipeline


def generate_img(prompt: str = "A flashy bottle that stands out from the rest."):
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,  # torch.float16,
        # variant="fp16",
        cache_dir="models",
    )
    pipe.to("cuda")
    # Play with these parameters to get different results
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=3.5).images[0]
    image.save("test.jpg")


generate_img()
