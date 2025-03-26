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


class MnemonicGen:
    def __init__(self):
        self.config = config.get("LLM")

        self.model_name = "Qwen/Qwen2.5-7B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir="models",
            # quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="models",
        )
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            }
        ]

    def generate_mnemonic(self, language: str = "Indonesian", word: str = "Kucing"):
        final_message = {
            "role": "user",
            "content": f"Think of a mnemonic to remember the {language} word {word}. Think of an English word or 2 words that sound similar to how Kucing would be pronounced in Indonesian. Also consider that the mnemonic should be an easy to imagine word and a word that is commonly used. Give a list of 10 mnemonic options based on these criteria.",
        }

        # For some reason using tokenizer.apply_chat_template() here causes weird output
        input = self.messages + [final_message]
        print(input)
        output = self.pipe(input)
        response = output[0]["generated_text"]
        logger.debug(f"Generated cue: {response}")
        return response


if __name__ == "__main__":
    mnemonic_gen = MnemonicGen()
    mnemonic = mnemonic_gen.generate_mnemonic()
