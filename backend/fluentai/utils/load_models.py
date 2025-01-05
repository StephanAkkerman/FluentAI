import gc
import os

import torch

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.services.imagine.image_gen import ImageGen
from fluentai.services.imagine.verbal_cue_gen import VerbalCue
from fluentai.services.mnemonic.phonetic.grapheme2phoneme import Grapheme2Phoneme
from fluentai.services.mnemonic.semantic.compute import SemanticSimilarity


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

    # LLM model
    llm_model = config.get("LLM").get("MODEL")
    if get_model_dir_name(llm_model) not in downloaded_models:
        logger.info(f"Downloading LLM model: {llm_model}")
        clean(VerbalCue()._initialize_pipe())

    # Image gen
    image_gen_model = config.get("IMAGE_GEN").get("LARGE_MODEL")
    if get_model_dir_name(image_gen_model) not in downloaded_models:
        logger.info(f"Downloading image gen model: {image_gen_model}")
        clean(ImageGen()._initialize_pipe())

    # Semantic similarity
    semantic_similarity_model = config.get("SEMANTIC_SIM").get("MODEL")
    if get_model_dir_name(semantic_similarity_model) not in downloaded_models:
        logger.info(
            f"Downloading semantic similarity model: {semantic_similarity_model}"
        )
        clean(SemanticSimilarity())


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
