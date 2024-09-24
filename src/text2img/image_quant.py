# First, in your terminal.
#
# $ python3 -m virtualenv env
# $ source env/bin/activate
# $ pip install torch torchvision transformers sentencepiece protobuf accelerate
# $ pip install git+https://github.com/huggingface/diffusers.git
# $ pip install optimum-quanto

import logging
import os

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from optimum.quanto import freeze, qfloat8, qint4, quantize
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("script.log", mode="w"),  # Log to a file
    ],
)

logger = logging.getLogger(__name__)


def main():
    try:
        logger.info("Script started.")

        # Set data type
        dtype = torch.bfloat16
        logger.info(f"Using data type: {dtype}")

        # Define repository and revision
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
        revision = "refs/pr/1"
        logger.info(f"Using repository: {bfl_repo}, revision: {revision}")

        # Load Scheduler
        logger.info("Loading FlowMatchEulerDiscreteScheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            bfl_repo, subfolder="scheduler", revision=revision, cache_dir="models"
        )
        logger.info("Scheduler loaded successfully.")

        # Load Text Encoder
        logger.info("Loading CLIPTextModel...")
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=dtype, cache_dir="models"
        )
        logger.info("CLIPTextModel loaded successfully.")

        # Load Tokenizer
        logger.info("Loading CLIPTokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=dtype, cache_dir="models"
        )
        logger.info("CLIPTokenizer loaded successfully.")

        # Load Second Tokenizer
        logger.info("Loading T5TokenizerFast...")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            bfl_repo,
            subfolder="tokenizer_2",
            torch_dtype=dtype,
            revision=revision,
            cache_dir="models",
        )
        logger.info("T5TokenizerFast loaded successfully.")

        # Load VAE
        logger.info("Loading AutoencoderKL...")
        vae = AutoencoderKL.from_pretrained(
            bfl_repo,
            subfolder="vae",
            torch_dtype=dtype,
            revision=revision,
            cache_dir="models",
        )
        logger.info("AutoencoderKL loaded successfully.")

        # Paths for saved models
        transformer_path = os.path.join("models", "transformer.pt")
        text_encoder_path = os.path.join("models", "text_encoder.pt")

        # Load Transformer
        if os.path.exists(transformer_path):
            logger.info(f"Loading saved Transformer from {transformer_path}...")
            transformer = torch.load(transformer_path)
            transformer.eval()
            logger.info("Transformer loaded successfully from saved file.")
        else:
            logger.info("Loading FluxTransformer2DModel from pretrained repository...")
            transformer = FluxTransformer2DModel.from_pretrained(
                bfl_repo,
                subfolder="transformer",
                torch_dtype=dtype,
                revision=revision,
                cache_dir="models",
            )
            logger.info("FluxTransformer2DModel loaded successfully.")

            # Quantize and Freeze Transformer
            logger.info("Quantizing Transformer with qint4 weights...")
            quantize(
                transformer,
                weights=qint4,
                exclude=["proj_out", "x_embedder", "norm_out", "context_embedder"],
            )
            logger.info("Transformer quantized successfully.")

            logger.info("Freezing Transformer...")
            freeze(transformer)
            logger.info("Transformer frozen successfully.")

            # Save Transformer
            logger.info(f"Saving Transformer to {transformer_path}...")
            torch.save(transformer, transformer_path)
            logger.info("Transformer saved successfully.")

        # Load Second Text Encoder
        if os.path.exists(text_encoder_path):
            logger.info(f"Loading saved T5EncoderModel from {text_encoder_path}...")
            text_encoder_2 = torch.load(text_encoder_path)
            text_encoder_2.eval()
            logger.info("T5EncoderModel loaded successfully from saved file.")
        else:
            logger.info("Loading T5EncoderModel from pretrained repository...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                bfl_repo,
                subfolder="text_encoder_2",
                torch_dtype=dtype,
                revision=revision,
                cache_dir="models",
            )
            logger.info("T5EncoderModel loaded successfully.")

            # Quantize and Freeze Second Text Encoder
            logger.info("Quantizing T5EncoderModel with qint4 weights...")
            quantize(text_encoder_2, weights=qint4)
            logger.info("T5EncoderModel quantized successfully.")

            logger.info("Freezing T5EncoderModel...")
            freeze(text_encoder_2)
            logger.info("T5EncoderModel frozen successfully.")

            # Save T5Encoder
            logger.info(f"Saving T5EncoderModel to {text_encoder_path}...")
            torch.save(text_encoder_2, text_encoder_path)
            logger.info("T5EncoderModel saved successfully.")

        # Create FluxPipeline
        logger.info("Creating FluxPipeline...")
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,  # Temporarily set to None
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,  # Temporarily set to None
        )
        logger.info("FluxPipeline created successfully.")

        # Assign quantized models to pipeline
        pipe.text_encoder_2 = text_encoder_2
        pipe.transformer = transformer
        logger.info("Assigned quantized models to FluxPipeline.")

        # Enable CPU offload
        # logger.info("Enabling model CPU offload...")
        # pipe.enable_model_cpu_offload()
        # logger.info("Model CPU offload enabled.")
        pipe = pipe.to("cuda")

        # Generate Image
        logger.info("Generating image...")
        generator = torch.Generator().manual_seed(12345)
        image = pipe(
            prompt="nekomusume cat girl, digital painting",
            width=1024,
            height=1024,
            num_inference_steps=4,
            generator=generator,
            guidance_scale=3.5,
        ).images[0]

        logger.info("Image generated successfully.")

        # Save Image
        image_path = "test_flux_distilled.png"
        logger.info(f"Saving image to {image_path}...")
        image.save(image_path)
        logger.info(f"Image saved successfully at {image_path}.")

        logger.info("Script completed successfully.")

    except Exception as e:
        logger.error("An error occurred during execution.", exc_info=True)


if __name__ == "__main__":
    # Print GPU information
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"Current GPU: {torch.cuda.current_device()}")

    main()
