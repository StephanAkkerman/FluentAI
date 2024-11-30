# To run: uvicorn api:app --reload
# if that doesn't work try: python -m uvicorn api:app --reload


from fastapi import FastAPI, HTTPException, Query

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Generator
import os
from constants.languages import G2P_LANGCODES

from fluentai.services.card_gen.main import generate_mnemonic_img

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


@app.post("/create_card/word_data", response_model=CreateCardResponse)
async def api_generate_mnemonic(request: CreateCardRequest):
    # Validate language code if necessary
    if request.language_code not in G2P_LANGCODES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    try:
        # Data placeholders
        data = {
            "IPA": "TODO",  # Replace with actual IPA generation logic
            "recording": "TODO"  # Replace with actual recording logic
        }

        # Return the StreamingResponse and metadata
        return {
            "IPA": data["IPA"],
            "recording": data["recording"],
        }

    except Exception as e:
        import logging
        logging.error(f"Error generating mnemonic: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.get("/create_card/img")
async def get_image(word: str = Query(...), language_code: str = Query(...)):
     # Validate language code if necessary
    if language_code not in G2P_LANGCODES:
        raise HTTPException(status_code=400, detail="Invalid language code")

    try:
        # Generate image and get its file path
        image_path = generate_mnemonic_img(word, language_code)
        print(image_path)

        # Ensure the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=500, detail="Generated image not found")

        # Return the image as a file response
        return FileResponse(image_path, media_type="image/jpeg")

    except Exception as e:
        import logging
        logging.error(f"Error generating mnemonic: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

