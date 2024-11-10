# To run: uvicorn api:app --reload
# if that doesn't work try: python -m uvicorn api:app --reload

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# from mnemonic.word2mnemonic import generate_mnemonic


app = FastAPI()

# Allow all origins for development (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],  # Replace "*" with your front-end URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Pydantic models for request and responses
class MnemonicRequest(BaseModel):
    word: str
    language_code: str


class MnemonicItem(BaseModel):
    token_ort: str
    distance: float
    imageability: float
    semantic_similarity: float
    orthographic_similarity: float
    score: float


class MnemonicResponse(BaseModel):
    items: List[MnemonicItem]


# Test version of generate_mnemonic
# Mock endpoint
@app.post("/create-card", response_model=MnemonicResponse)
def mock_generate_mnemonic():
    """
    Generate a mnemonic card for a given word.

    Returns
    -------
    MnemonicResponse
        _description_
    """
    # Mock data to simulate the response
    mock_data = [
        {
            "token_ort": "example1",
            "distance": 0.1,
            "imageability": 0.8,
            "semantic_similarity": 0.7,
            "orthographic_similarity": 0.9,
            "score": 0.85,
        },
        {
            "token_ort": "example2",
            "distance": 0.2,
            "imageability": 0.7,
            "semantic_similarity": 0.6,
            "orthographic_similarity": 0.8,
            "score": 0.75,
        },
    ]

    mnemonic_items = [MnemonicItem(**item) for item in mock_data]
    return MnemonicResponse(items=mnemonic_items)


# @app.post("/generate_mnemonic", response_model=MnemonicResponse)
# def api_generate_mnemonic(request: MnemonicRequest):
#     # Validate language code if necessary
#     # (Assuming you have access to G2P_LANGCODES in your api.py)
#     from constants.languages import G2P_LANGCODES
#     if request.language_code not in G2P_LANGCODES:
#         raise HTTPException(status_code=400, detail="Invalid language code")

#     try:
#         # Call your existing function
#         top = generate_mnemonic(request.word, request.language_code)

#         if top is None or top.empty:
#             raise HTTPException(status_code=404, detail="No mnemonics found")

#         # Convert DataFrame to list of dictionaries
#         top_dict = top.to_dict(orient='records')

#         # Convert dictionaries to MnemonicItem instances
#         mnemonic_items = [MnemonicItem(**item) for item in top_dict]

#         return MnemonicResponse(items=mnemonic_items)

#     except Exception as e:
#         # Log the exception if necessary
#         import logging
#         logging.error(f"Error generating mnemonic: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")
