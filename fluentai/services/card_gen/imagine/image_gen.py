import gc
import os
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image, SanaPipeline

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


class ImageGen:
    def __init__(self):
        # Get all config values
        self.model = config.get("IMAGE_GEN", {}).get("MODEL", "stabilityai/sdxl-turbo")
        self.model_name = self.model.split("/")[-1]

        self.output_dir = Path(
            config.get("IMAGE_GEN", {}).get("OUTPUT_DIR", "output")
        ).resolve()
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_gen_params = config.get("IMAGE_GEN", {}).get("PARAMS", {})

        pipe_func = self._get_pipe_func()
        self.pipe = pipe_func.from_pretrained(
            self.model,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="models",
        )

    def _get_pipe_func(self):
        if "sana" in self.model_name.lower():
            return SanaPipeline
        else:
            return AutoPipelineForText2Image

    def generate_img(
        self,
        prompt: str = "A flashy bottle that stands out from the other bottles.",
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

        # Check if cuda is enabled
        if torch.cuda.is_available():
            logger.debug("CUDA is available. Moving the t2i pipeline to CUDA.")
            self.pipe.to("cuda")
        else:
            logger.info("CUDA is not available. Running the image gen pipeline on CPU.")

        logger.info(f"Generating image for prompt: {prompt}")

        image = self.pipe(prompt=prompt, **self.image_gen_params).images[0]

        logger.info(f"Saving image to: {file_path}")

        image.save(file_path)

        if config.get("IMAGE_GEN", {}).get("DELETE_AFTER_USE", True):
            logger.debug("Deleting the VerbalCue model to free up memory.")
            del self.pipe
            gc.collect()
            torch.cuda.empty_cache()

        return file_path


if __name__ == "__main__":
    ImageGen().generate_img()
