from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from fluentai.constants.config import config
from fluentai.logger import logger
from fluentai.utils.model_mem import manage_memory


class VerbalCue:
    def __init__(self, model_name: str = None):
        self.config = config.get("LLM")
        self.offload = self.config.get("OFFLOAD")
        self.model_name = model_name if model_name else self.config.get("MODEL")

        self.generation_args = {
            "max_new_tokens": 256,
            "do_sample": True,
            "return_full_text": False,
            "temperature": config.get("LLM").get("temperature"),
        }
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for making mnemonics. You will help users create a mnemonic by generating a cue that connects two words, keep it short and catchy!",
            },
            {
                "role": "user",
                "content": "Generate a mnemonic sentence for the given input. Start the sentence with 'imagine' and keep it simple. \n Input: English Word: bottle | Mnemonic Word: flashy",
            },
            {
                "role": "assistant",
                "content": "Imagine a flashy bottle that stands out from the rest.",
            },
        ]
        # This will be initialized later
        self.pipe = None
        self.tokenizer = None  # Initialize tokenizer attribute
        self.model = None  # Initialize model attribute

    def _initialize_pipe(self):
        """Initialize the pipeline."""
        logger.debug(f"Initializing pipeline for LLM with model: {self.model_name}")

        bnb_config = None
        if config.get("LLM").get("QUANTIZATION") == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif config.get("LLM").get("QUANTIZATION") == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda" if self.offload else "auto",
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir="models",
            quantization_config=bnb_config,
        )

        # Check if LoRA should be enabled
        if config.get("LLM").get("USE_LORA"):
            # Load the model with LoRA
            self.model = PeftModel.from_pretrained(
                self.model, config.get("LLM").get("LORA")
            )
            self.model = self.model.merge_and_unload()

        # Ensure the model is in evaluation mode (disables dropout, etc.)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="models",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @manage_memory(
        targets=["model"], delete_attrs=["model", "pipe", "tokenizer"], move_kwargs={}
    )
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
