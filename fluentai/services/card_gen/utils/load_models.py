import gc
import os

import torch
from diffusers import AutoPipelineForText2Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


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
    return f"models--{model.replace('/', '--')}"


def download_all_models():
    """Download all models from the Hugging Face Hub and clean them up after loading."""
    # Get the directory names in /models
    downloaded_models = [
        str(entry)
        for entry in os.listdir("models")
        if os.path.isdir(os.path.join("models", entry)) and entry.startswith("models--")
    ]

    # G2P model
    g2p_model = config.get("G2P").get("MODEL")
    if get_model_dir_name(g2p_model) not in downloaded_models:
        logger.info(f"Downloading G2P model: {g2p_model}")
        clean(T5ForConditionalGeneration.from_pretrained(g2p_model, cache_dir="models"))

    # G2P tokenizer
    g2p_tokenizer = config.get("G2P").get("TOKENIZER")
    if get_model_dir_name(g2p_tokenizer) not in downloaded_models:
        logger.info(f"Downloading G2P tokenizer: {g2p_tokenizer}")
        clean(AutoTokenizer.from_pretrained(g2p_tokenizer, cache_dir="models"))

    # LLM model
    llm_model = config.get("LLM").get("MODEL")
    if get_model_dir_name(llm_model) not in downloaded_models:
        logger.info(f"Downloading LLM model: {llm_model}")
        clean(AutoModelForCausalLM.from_pretrained(llm_model, cache_dir="models"))

    # LLM tokenizer
    llm_tokenizer = config.get("LLM").get("TOKENIZER")
    if get_model_dir_name(llm_tokenizer) not in downloaded_models:
        logger.info(f"Downloading LLM tokenizer: {llm_tokenizer}")
        clean(AutoTokenizer.from_pretrained(llm_tokenizer, cache_dir="models"))

    # Image gen
    image_gen_model = config.get("IMAGE_GEN").get("MODEL")
    if get_model_dir_name(image_gen_model) not in downloaded_models:
        logger.info(f"Downloading image gen model: {image_gen_model}")
        clean(
            AutoPipelineForText2Image.from_pretrained(
                image_gen_model, cache_dir="models"
            )
        )

    # Semantic similarity
    semantic_similarity_model = config.get("SEMANTIC_SIM").get("MODEL")
    if get_model_dir_name(semantic_similarity_model) not in downloaded_models:
        logger.info(
            f"Downloading semantic similarity model: {semantic_similarity_model}"
        )
        clean(
            SentenceTransformer(
                semantic_similarity_model, trust_remote_code=True, cache_folder="models"
            )
        )


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
