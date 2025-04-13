import base64
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mnemorai.constants.config import config
from mnemorai.constants.languages import G2P_LANGCODES, G2P_LANGUAGES
from mnemorai.logger import logger
from mnemorai.run import MnemonicPipeline

create_card_router = APIRouter()


# Define Pydantic models for request and responses
class CreateCardRequest(BaseModel):
    word: str
    language_code: str


class CreateCardResponse(BaseModel):
    IPA: str = None  # Placeholder for future implementation
    recording: str = None  # Placeholder for future implementation


@create_card_router.post("/create_card/word_data", response_model=CreateCardResponse)
async def api_generate_mnemonic(request: CreateCardRequest) -> dict:
    """
    Calls the main function to generate a mnemonic for a given word and language code.

    Parameters
    ----------
    request : CreateCardRequest
        The request object containing the word and language code.

    Returns
    -------
    dict
        The response object containing the IPA and recording of the generated mnemonic.

    Raises
    ------
    HTTPException
        If the language code is invalid.
    HTTPException
        If an error occurs during the generation process.
    """
    # Validate language code if necessary
    if request.language_code not in G2P_LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    try:
        # Data placeholders
        data = {
            "IPA": "TODO",  # Replace with actual IPA generation logic
            "recording": "TODO",  # Replace with actual recording logic
        }

        # Return the StreamingResponse and metadata
        return {
            "IPA": data["IPA"],
            "recording": data["recording"],
        }

    except Exception as e:
        logger.error(f"Error generating mnemonic: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@create_card_router.get("/create_card/img")
async def get_image(
    word: str = Query(...),
    language_code: str = Query(...),
    llm_model: str = Query(None),
    image_model: str = Query(None),
    keyword: str = Query(None),
    key_sentence: str = Query(None),
) -> JSONResponse:
    """
    Generates a mnemonic image for a given word and language code.

    Parameters
    ----------
    word : str
        The word to generate a mnemonic image for.
    language_code : str
        The language code of the word.
    llm_model : str, optional
        The name of the LLM model to use for verbal cue generation.
    image_model : str, optional
        The name of the image model to use for image generation.
    keyword : str, optional
        A user-supplied keyword to use in the mnemonic.
    key_sentence : str, optional
        A user-supplied key sentence to use as the prompt for image generation.

    Returns
    -------
    JSONResponse
        A JSON response containing the generated image, verbal cue, translation,
        TTS file path, and IPA.

    Raises
    ------
    HTTPException
        If the language code is invalid, the generated image is not found,
        or an error occurs during the generation process.
    """
    if language_code not in G2P_LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    mnemonic_pipe = MnemonicPipeline()

    try:
        image_path, verbal_cue, translation, tts_path, ipa = (
            await mnemonic_pipe.generate_mnemonic_img(
                word, language_code, llm_model, image_model, keyword, key_sentence
            )
        )

        if not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Generated image not found")

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        with open(tts_path, "rb") as tts_file:
            tts_bytes = tts_file.read()

        return JSONResponse(
            content={
                "image": base64.b64encode(image_bytes).decode("utf-8"),
                "verbal_cue": verbal_cue,
                "translation": translation.title(),
                "tts_file": base64.b64encode(tts_bytes).decode("utf-8"),
                "ipa": ipa,
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating mnemonic: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@create_card_router.get("/create_card/supported_languages")
async def get_supported_languages() -> JSONResponse:
    """
    Returns a list of languages that the backend supports.

    Returns
    -------
    JSONResponse
        The list of supported languages
    """
    return JSONResponse(content={"languages": G2P_LANGCODES})


@create_card_router.get("/create_card/image_models")
async def get_image_models() -> JSONResponse:
    """
    Returns a list of available image generation models, with the recommended model at the top.
    """
    image_gen_config = config.get("IMAGE_GEN", {})
    recommended_model = image_gen_config.get("LARGE_MODEL")
    models = {
        "large": image_gen_config.get("LARGE_MODEL"),
        "medium": image_gen_config.get("MEDIUM_MODEL"),
        "small": image_gen_config.get("SMALL_MODEL"),
        "tiny": image_gen_config.get("TINY_MODEL"),
    }

    # Filter out None values and sort with recommended model first
    available_models = [model for model in models.values() if model]
    available_models.sort(key=lambda x: x != recommended_model)

    return JSONResponse(content={"models": available_models})


@create_card_router.get("/create_card/llm_models")
async def get_llm_models() -> JSONResponse:
    """
    Returns a list of available LLM models, with the recommended model at the top.
    """
    llm_config = config.get("LLM")
    recommended_model = llm_config.get("MODEL")
    models = {
        "recommended": recommended_model,
        "all": [
            model
            for model in llm_config.values()
            if isinstance(model, str) and model != recommended_model
        ],
    }

    # Combine and sort with recommended model first
    available_models = [recommended_model] + models["all"]

    return JSONResponse(content={"models": available_models})
