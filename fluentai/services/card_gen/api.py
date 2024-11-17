# To run: uvicorn api:app --reload
# if that doesn't work try: python -m uvicorn api:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Generator
import os

from main import generate_mnemonic_img

app = FastAPI()

# Allow all origins for development (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://akkerman.ai/FluentAI/"
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


@app.post("/create_card", response_model=CreateCardResponse)
async def api_generate_mnemonic(request: CreateCardRequest):
    # Validate language code if necessary
    from constants.languages import G2P_LANGCODES
    if request.language_code not in G2P_LANGCODES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    try:
        # Generate image and get its file path
        image_path = generate_mnemonic_img(request.word, request.language_code)

        # Ensure the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Generated image not found")

        # Define a generator for streaming the image
        def image_stream() -> Generator[bytes, None, None]:
            with open(image_path, "rb") as image_file:
                while chunk := image_file.read(1024):  # Stream in chunks of 1 KB
                    yield chunk

        # Metadata placeholders
        metadata = {
            "IPA": "TODO",  # Replace with actual IPA generation logic
            "recording": "TODO"  # Replace with actual recording logic
        }

        # Return the StreamingResponse and metadata
        return {
            "image": StreamingResponse(image_stream(), media_type="image/jpeg"),
            "IPA": metadata["IPA"],
            "recording": metadata["recording"],
        }

    except Exception as e:
        import logging
        logging.error(f"Error generating mnemonic: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")