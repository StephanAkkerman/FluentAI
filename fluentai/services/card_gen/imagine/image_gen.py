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
    ).to("cuda")

    logger.info(f"Generating image for prompt: {prompt}")

    # Get the parameters for image generation from config
    image_gen_params = config.get("IMAGE_GEN", {}).get("PARAMS", {})
    image = pipe(prompt=prompt, **image_gen_params).images[0]
    
    file_path = f"img/text2img_tests/{model_name}_{word1}-{word2}.jpg"


    image.save(file_path)
    logger.info("Generated image!")
    return file_path


if __name__ == "__main__":
    generate_img()
