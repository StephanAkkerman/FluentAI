import os
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)
from transformers import AutoModel
from transformers import BitsAndBytesConfig as BitsAndBytesConfig

from mnemorai.constants.config import config
from mnemorai.logger import logger
from mnemorai.utils.model_mem import manage_memory


class ImageGen:
    def __init__(self, model: str = None):
        self.config = config.get("IMAGE_GEN", {})
        self.offload = self.config.get("OFFLOAD")

        # Select model based on VRAM or provided model
        if model:
            self.model = model
            logger.debug(f"Using provided image model: {self.model}")
        else:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory
            else:
                vram = 0
            self._select_model(vram)
            logger.debug(f"Selected image model based on VRAM: {self.model}")

        self.model_name = self.model.split("/")[-1]
        self.output_dir = Path(self.config.get("OUTPUT_DIR", "output")).resolve()
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_gen_params = self.config.get("PARAMS", {})

        # Initialize pipe to None; will be loaded on first use
        self.pipe = None

    def _select_model(self, vram):
        """Select the appropriate model based on available VRAM."""
        if vram > 9e9:
            self.model = self.config.get("LARGE_MODEL", "stabilityai/sdxl-turbo")
        elif vram > 6e9:
            self.model = self.config.get("MEDIUM_MODEL", "stabilityai/sdxl-turbo")
        elif vram > 3e9:
            self.model = self.config.get("SMALL_MODEL", "stabilityai/sdxl-turbo")
        else:
            raise RuntimeError(
                "Not enough VRAM available for image generation. Please run on a machine with a GPU."
            )

    def _get_pipe_func(self):
        if "sana" in self.model_name.lower():
            return SanaPipeline
        else:
            return AutoPipelineForText2Image

    def _initialize_pipe(self):
        """Initialize the pipeline."""
        pipe_func = self._get_pipe_func()
        logger.debug(f"Initializing pipeline for model: {self.model}")

        quantization = self.config.get("QUANTIZATION")

        if quantization != "4bit" and quantization != "8bit":
            logger.debug("Using default model loading without quantization")
            self.pipe = pipe_func.from_pretrained(
                self.model,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir="models",
            )
        else:
            if "sana" in self.model_name.lower():
                if quantization == "8bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.debug("Using 8-bit quantization for Sana model")
                elif quantization == "4bit":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True)
                    logger.debug("Using 4-bit quantization for Sana model")
                else:
                    raise ValueError(
                        f"Invalid quantization type. Use '8bit' or '4bit'. Your quantization is: {quantization}"
                    )

                text_encoder_8bit = AutoModel.from_pretrained(
                    self.model,
                    subfolder="text_encoder",
                    quantization_config=quant_config,
                    torch_dtype=torch.float16,
                    cache_dir="models",
                )

                quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
                transformer_8bit = SanaTransformer2DModel.from_pretrained(
                    self.model,
                    subfolder="transformer",
                    quantization_config=quant_config,
                    torch_dtype=torch.float16,
                    cache_dir="models",
                )

                self.pipe = SanaPipeline.from_pretrained(
                    self.model,
                    text_encoder=text_encoder_8bit,
                    transformer=transformer_8bit,
                    torch_dtype=torch.float16,
                    device_map="balanced",
                )
            else:
                raise NotImplementedError("Quantization not supported for this model.")

    @manage_memory(
        targets=["pipe"],
        delete_attrs=["pipe"],
        move_kwargs={"silence_dtype_warnings": True},
    )
    def generate_img(
        self,
        prompt: str = "Imagine a flashy bottle that stands out from the other bottles.",
        word1: str = "flashy",
        word2: str = "bottle",
    ):
        """
        Generate an image from a text prompt using a text-to-image model.

        Parameters
        ----------
        prompt : str, optional
            The prompt to give to the model, by default "A flashy bottle that stands out from the other bottles."
        word1 : str, optional
            The first word of mnemonic, by default "flashy"
        word2 : str, optional
            The second word, by default "bottle"
        """
        file_path = self.output_dir / f"{word1}_{word2}_{self.model_name}.png"

        # Clean prompt by dropping "imagine " prefix
        prompt = prompt.lower().lstrip("imagine").strip()

        logger.info(f"Generating image for prompt: {prompt}")
        image = self.pipe(prompt=prompt, **self.image_gen_params).images[0]
        logger.info(f"Saving image to: {file_path}")

        image.save(file_path)

        return file_path


if __name__ == "__main__":
    img_gen = ImageGen()
    img_gen.generate_img()
    img_gen.generate_img("Imagine a cat that walks over the moon", "cat", "moon")
