import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mnemorai.api.routes.anki import anki_router
from mnemorai.api.routes.create_card import create_card_router
from mnemorai.utils.load_models import download_all_models

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://akkerman.ai",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anki_router)
app.include_router(create_card_router)


def main():
    """Start the FastAPI application."""
    # Start by downloading all models
    download_all_models()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Hosting default: 127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
