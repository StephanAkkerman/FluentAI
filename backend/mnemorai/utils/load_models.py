import gc
import os

import torch

from mnemorai.constants.config import config
from mnemorai.logger import logger
from mnemorai.services.imagine.image_gen import ImageGen
from mnemorai.services.imagine.verbal_cue_gen import VerbalCue
from mnemorai.services.pre.grapheme2phoneme import Grapheme2Phoneme


def get_model_dir_name(model: str) -> str:
    """Get the directory name for the model.

    Parameters
    ----------
    model : str
        The model name

    Returns
    -------
    str
        The directory name for the model
    """
    # If there is no slash in the model name, append "sentence-transformers" to the model name
    if "/" not in model:
        return f"models--sentence-transformers--{model.lower()}"
    return f"models--{model.replace('/', '--')}"


def download_all_models():
    """Download all models from the Hugging Face Hub and clean them up after loading."""
    # Make the models dir if it does not exist yet
    os.makedirs("models", exist_ok=True)

    # Get the directory names in /models
    downloaded_models = [
        str(entry)
        for entry in os.listdir("models")
        if os.path.isdir(os.path.join("models", entry)) and entry.startswith("models--")
    ]

    # G2P model & tokenizer
    g2p_model = config.get("G2P").get("MODEL")
    if get_model_dir_name(g2p_model) not in downloaded_models:
        logger.info(f"Downloading G2P model: {g2p_model}")
        clean(Grapheme2Phoneme())

    # Use vram to select the model
    vram = 0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
    if vram == 0:
        raise RuntimeError("No GPU available. Please run on a machine with a GPU.")

    # Default to small
    model = "SMALL_MODEL"
    if vram > 9e9:
        model = "LARGE_MODEL"
    elif vram > 6e9:
        model = "MEDIUM_MODEL"
    elif vram > 3e9:
        model = "SMALL_MODEL"

    # LLM model
    llm_model = config.get("LLM").get(model)
    if get_model_dir_name(llm_model) not in downloaded_models:
        logger.info(f"Downloading LLM model: {llm_model}")
        clean(VerbalCue()._initialize_pipe())

    # Image gen
    image_gen_model = config.get("IMAGE_GEN").get(model)
    if get_model_dir_name(image_gen_model) not in downloaded_models:
        logger.info(f"Downloading image gen model: {image_gen_model}")
        clean(ImageGen()._initialize_pipe())


def clean(var):
    """Clean up the model and free up memory.

    Parameters
    ----------
    var : object
        The model to clean up
    """
    del var
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    download_all_models()
