# To run: uvicorn api:app --reload
# if that doesn't work try: python -m uvicorn api:app --reload

import argparse
import base64
import os

import uvicorn
from constants.languages import G2P_LANGCODES, G2P_LANGUAGES
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fluentai.services.card_gen.main import generate_mnemonic_img
from fluentai.services.card_gen.utils.logger import logger

app = FastAPI()

# Allow all origins for development (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://akkerman.ai",
    ],  # Replace "*" with your front-end URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Pydantic models for request and responses
class CreateCardRequest(BaseModel):
    word: str
    language_code: str


class CreateCardResponse(BaseModel):
    IPA: str = None  # Placeholder for future implementation
    recording: str = None  # Placeholder for future implementation


@app.post("/create_card/word_data", response_model=CreateCardResponse)
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
    if request.language_code not in G2P_LANGCODES:
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


@app.get("/create_card/img")
async def get_image(
    word: str = Query(...), language_code: str = Query(...)
) -> JSONResponse:
    """
    Generates a mnemonic image for a given word and language code.

    Parameters
    ----------
    word : str, optional
        The word to generate a mnemonic image for, by default Query(...)
    language_code : str, optional
        The language code of the word, by default Query(...)

    Returns
    -------
    FileResponse
        The image file response.

    Raises
    ------
    HTTPException
        If the language code is invalid.
    HTTPException
        If the generated image is not found.
    HTTPException
        If an error occurs during the generation process.
    """
    # Validate language code if necessary
    if language_code not in G2P_LANGCODES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    try:
        # Generate image and get its file path along with verbal cue and translation
        image_path, verbal_cue, translation, tts_path, ipa = generate_mnemonic_img(
            word, language_code
        )

        # Ensure the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Generated image not found")

        # Read the image file as bytes
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Read the image file as bytes
        with open(tts_path, "rb") as tts_file:
            tts_bytes = tts_file.read()

        # Return a JSON response with image and metadata
        return JSONResponse(
            content={
                "image": base64.b64encode(image_bytes).decode("utf-8"),
                "verbal_cue": verbal_cue,
                "translation": translation.title(),
                "tts_file": base64.b64encode(tts_bytes).decode("utf-8"),
                "ipa": ipa,
            }
        )
    except Exception as e:
        logger.error(f"Error generating mnemonic: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/create_card/supported_languages")
async def get_supported_languages() -> JSONResponse:
    """
    Returns a list of languages that the backend supports.

    Returns
    -------
    JSONResponse
        The list of supported languages
    """
    return JSONResponse(content={"languages": G2P_LANGUAGES})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Hosting default: 127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    uvicorn.run("api:app", host=args.host, port=args.port)
