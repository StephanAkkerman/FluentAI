import gc
import os
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


def generate_img(
    prompt: str = "A flashy bottle that stands out from the other bottles.",
    word1: str = "flashy",
    word2: str = "bottle",
):
    """
    Generate an image from a text prompt using a text-to-image model.

    Parameters
    ----------
    prompt : str, optional
        The prompt to give to the model, by default "A flashy bottle that stands out from the other bottles."
    word1 : str, optional
        The first word of mnemonic, by default "flashy"
    word2 : str, optional
        The second word, by default "bottle"
    """
    model = config.get("IMAGE_GEN", {}).get("MODEL", "stabilityai/sdxl-turbo")
    model_name = model.split("/")[-1]

    pipe = AutoPipelineForText2Image.from_pretrained(
        model,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir="models",
    )

    # Check if cuda is enabled
    if torch.cuda.is_available():
        logger.debug("CUDA is available. Moving the t2i pipeline to CUDA.")
        pipe.to("cuda")
    else:
        logger.info("CUDA is not available. Running the image gen pipeline on CPU.")

    logger.info(f"Generating image for prompt: {prompt}")

    # Get the parameters for image generation from config
    image_gen_params = config.get("IMAGE_GEN", {}).get("PARAMS", {})
    image = pipe(prompt=prompt, **image_gen_params).images[0]

    # Get the output directory from config
    output_dir = Path(config.get("IMAGE_GEN", {}).get("OUTPUT_DIR", "output")).resolve()
    os.makedirs(output_dir, exist_ok=True)

    file_path = output_dir / f"{word1}_{word2}_{model_name}.png"
    logger.info(f"Saving image to: {file_path}")

    image.save(file_path)

    if config.get("IMAGE_GEN", {}).get("DELETE_AFTER_USE", True):
        logger.debug("Deleting the VerbalCue model to free up memory.")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return file_path


if __name__ == "__main__":
    generate_img()
