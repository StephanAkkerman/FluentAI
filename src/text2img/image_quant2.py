import logging
import os

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, qint4, quantize
from transformers import CLIPTextModel, T5EncoderModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("flux_pipeline.log", mode="w"),  # Log to a file
    ],
)

logger = logging.getLogger(__name__)


def load_or_build_model(model_name, build_func, save_path, load_func):
    """
    Load a model from disk if it exists; otherwise, build it using build_func, save it, and return it.

    Args:
        model_name (str): Name of the model for logging purposes.
        build_func (callable): Function to build the model.
        save_path (str): Path to save the model.
        load_func (callable): Function to load the model.

    Returns:
        The loaded or newly built model.
    """
    if os.path.exists(save_path):
        logger.info(f"Loading existing {model_name} from {save_path}...")
        model = load_func(save_path)
        logger.info(f"{model_name} loaded successfully from disk.")
    else:
        logger.info(f"{model_name} not found. Building {model_name}...")
        model = build_func()
        logger.info(f"{model_name} built successfully. Saving to {save_path}...")
        torch.save(model, save_path)
        logger.info(f"{model_name} saved successfully.")
    return model


def main():
    try:
        logger.info("Script started.")

        # Define repository and data type
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
        dtype = torch.bfloat16
        logger.info(f"Using repository: {bfl_repo} with data type: {dtype}")

        # Define model save paths
        transformer_save_path = os.path.join("models", "transformer.pt")
        text_encoder_save_path = os.path.join("models", "text_encoder_2.pt")

        # Ensure the models directory exists
        os.makedirs("models", exist_ok=True)

        # Define functions to build models
        def build_transformer():
            logger.info("Loading FluxTransformer2DModel from URL...")
            transformer = FluxTransformer2DModel.from_single_file(
                "https://huggingface.co/argmaxinc/mlx-FLUX.1-schnell-4bit-quantized/blob/main/flux-schnell-4bit-quantized.safetensors",
                torch_dtype=dtype,
                cache_dir="models",
            )
            logger.info("FluxTransformer2DModel loaded from URL successfully.")
            logger.info("Quantizing Transformer with qfloat8 weights...")
            quantize(transformer, weights=qint4)
            logger.info("Transformer quantized successfully.")
            logger.info("Freezing Transformer...")
            freeze(transformer)
            logger.info("Transformer frozen successfully.")
            return transformer

        def build_text_encoder():
            logger.info("Loading T5EncoderModel from repository...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                bfl_repo,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                cache_dir="models",
            )
            logger.info("T5EncoderModel loaded successfully.")
            logger.info("Quantizing T5EncoderModel with qfloat8 weights...")
            quantize(text_encoder_2, weights=qfloat8)
            logger.info("T5EncoderModel quantized successfully.")
            logger.info("Freezing T5EncoderModel...")
            freeze(text_encoder_2)
            logger.info("T5EncoderModel frozen successfully.")
            return text_encoder_2

        # Load or build Transformer
        transformer = load_or_build_model(
            model_name="Trans",
            build_func=build_transformer,
            save_path=transformer_save_path,
            load_func=lambda path: torch.load(path, map_location="cpu"),
        )

        # Load or build Text Encoder
        text_encoder_2 = load_or_build_model(
            model_name="T5EncoderModel",
            build_func=build_text_encoder,
            save_path=text_encoder_save_path,
            load_func=lambda path: torch.load(path, map_location="cpu"),
        )

        # Create FluxPipeline
        logger.info("Creating FluxPipeline...")
        pipe = FluxPipeline.from_pretrained(
            bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype
        )
        logger.info("FluxPipeline created successfully.")

        # Assign loaded models to pipeline
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2
        logger.info("Assigned Transformer and T5EncoderModel to FluxPipeline.")

        # Enable CPU offload
        logger.info("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
        logger.info("Model CPU offload enabled.")

        # Define prompt and generation parameters
        prompt = "A cat holding a sign that says hello world"
        logger.info(f"Generating image with prompt: '{prompt}'")

        # Generate Image
        generator = torch.Generator("cpu").manual_seed(0)
        image = pipe(
            prompt,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=20,
            generator=generator,
        ).images[0]
        logger.info("Image generated successfully.")

        # Save Image
        image_path = "flux-fp4-dev.png"
        logger.info(f"Saving image to {image_path}...")
        image.save(image_path)
        logger.info(f"Image saved successfully at {image_path}.")

        logger.info("Script completed successfully.")

    except Exception as e:
        logger.error("An error occurred during execution.", exc_info=True)


if __name__ == "__main__":
    main()
