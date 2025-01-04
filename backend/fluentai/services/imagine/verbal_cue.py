import functools
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from fluentai.services.card_gen.constants.config import config
from fluentai.services.card_gen.utils.logger import logger


def manage_model_memory(method):
    """
    Decorator to manage model memory by offloading to GPU before the method call.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Initialize the pipe if it's not already loaded
        if self.pipe is None:
            self._initialize_pipe()

        # Move to GPU if offloading is enabled
        if self.offload:
            logger.debug("Moving the model to GPU (cuda).")
            self.model.to("cuda")

        try:
            # Execute the decorated method
            result = method(self, *args, **kwargs)
        finally:
            # Delete the pipeline if DELETE_AFTER_USE is True
            if self.config.get("DELETE_AFTER_USE", True):
                logger.debug("Deleting the model to free up memory.")
                del self.model
                del self.pipe
                del self.tokenizer
                self.model = None
                self.pipe = None
                self.tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()

            # Move the pipeline back to CPU if offloading is enabled
            if self.offload and self.pipe is not None:
                logger.debug("Moving the model back to CPU.")
                self.model.to("cpu")

        return result

    return wrapper


class VerbalCue:
    def __init__(self, model_name: str = None):
        self.config = config.get("LLM")
        self.offload = self.config.get("OFFLOAD")
        self.model_name = model_name if model_name else self.config.get("MODEL")

        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 1.0,
        }
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for making mnemonics. You will help users create a mnemonic by generating a cue that connects two words, keep it short and catchy!",
            },
            {
                "role": "user",
                "content": "Write a short, catchy sentence that connects flashy and bottle. Also, the sentence must start with 'Imagine'.",
            },
            {
                "role": "assistant",
                "content": "Imagine a flashy bottle that stands out from the rest!",
            },
        ]
        # This will be initialized later
        self.pipe = None
        self.tokenizer = None  # Initialize tokenizer attribute
        self.model = None  # Initialize model attribute

    def _initialize_pipe(self):
        """Initialize the pipeline."""
        logger.debug(f"Initializing pipeline for LLM with model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda" if self.offload else "auto",
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir="models",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="models",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @manage_model_memory
    def generate_cue(self, word1: str, word2: str) -> str:
        """
        Generate a verbal cue that connects two words.

        Parameters
        ----------
        word1 : str
            The first word.
        word2 : str
            The second word.

        Returns
        -------
        str
            The generated verbal cue.
        """
        final_message = {
            "role": "user",
            "content": f"Write a short, catchy sentence that connects {word1} and {word2}. Also, the sentence must start with 'Imagine'. ",
        }
        output = self.pipe(self.messages + [final_message], **self.generation_args)
        response = output[0]["generated_text"]
        logger.debug(f"Generated cue: {response}")

        return response


if __name__ == "__main__":
    vc = VerbalCue()
    print(vc.generate_cue("hairdresser ", "freezer"))
    print(vc.generate_cue("needing ", "broken"))
