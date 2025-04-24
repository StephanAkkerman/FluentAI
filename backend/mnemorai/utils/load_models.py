import gc
import os

import torch

from mnemorai.constants.config import config
from mnemorai.logger import logger


def select_model(config: dict) -> str:
    """
    Pick the heaviest model that fits in available_vram_bytes, based on config.

    Returns
    -------
    str
        The name of the selected model
    """
    available_vram_bytes = 0
    if torch.cuda.is_available():
        available_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    else:
        raise RuntimeError("No GPU vram found.")

    if not isinstance(config, dict):
        raise RuntimeError("Invalid config: section missing or not a dict")

    # build (required_bytes, model_name) list
    candidates = []
    for key, info in config.items():
        try:
            gb = float(info["VRAM"])
            name = str(info["NAME"])
        except (KeyError, TypeError, ValueError):
            # skip entries that lack VRAM/NAME or are malformed
            continue
        # Multiply by 1e9
        candidates.append((int(gb * 1e9), name))

    if not candidates:
        raise RuntimeError(
            "No valid IMAGE_GEN models found in config; "
            "each entry needs a NAME (str) and VRAM (number)"
        )

    # sort descending by VRAM requirement
    candidates.sort(key=lambda x: x[0], reverse=True)

    # pick the first that fits
    for required, name in candidates:
        if available_vram_bytes >= required:
            return name

    # if nothing fits, report the smallest requirement so user knows the bar
    smallest = candidates[-1][0]
    raise RuntimeError(
        f"Not enough VRAM ({available_vram_bytes} bytes) for any model; "
        f"need at least {smallest} bytes."
    )


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
        from mnemorai.services.pre.grapheme2phoneme import Grapheme2Phoneme

        logger.info(f"Downloading G2P model: {g2p_model}")
        clean(Grapheme2Phoneme())

    # LLM model
    llm_model = select_model(config.get("LLM"))
    if get_model_dir_name(llm_model) in downloaded_models:
        from mnemorai.services.imagine.verbal_cue_gen import VerbalCue

        logger.info(f"Downloading LLM model: {llm_model}")
        clean(VerbalCue()._initialize_pipe())

    # Image gen
    image_gen_model = select_model(config.get("IMAGE_GEN"))
    if get_model_dir_name(image_gen_model) not in downloaded_models:
        from mnemorai.services.imagine.image_gen import ImageGen

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
