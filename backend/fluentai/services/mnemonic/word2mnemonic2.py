from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from fluentai.constants.config import config
from fluentai.logger import logger


class MnemonicGen:
    def __init__(self):
        self.config = config.get("LLM")

        self.model_name = (
            "microsoft/Phi-3-mini-4k-instruct"
            # "Qwen/Qwen2.5-7B-Instruct"
        )

        # use recommended settings for phi-3.5
        self.generation_args = {
            "max_new_tokens": 512,
            "do_sample": True,
            "num_beams": 1,
            "top_k": 50,
            "top_p": 0.95,
            "return_full_text": False,
            "temperature": 0.6,
        }

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
                "content": "You are a helpful assistant that generates mnemonics for foreign language vocabulary words. The mnemonic should sound very similar to the foreign word.  Use common English words or short phrases.",
            }
        ]

    def generate_mnemonic(self, language: str = "Indonesian", word: str = "Kucing"):
        final_message = {
            "role": "user",
            "content": f"""Think of a mnemonic to remember the {language} word {word}. 
Think of an English word or a combination of 2 words that sound similar to how {word} would be pronounced in {language}. 
Also consider that the mnemonic should be an easy to imagine word and a word that is commonly used. 
Do not simply translate the word, the mnemonic should be a *memory aid* based on sound, not a translation.
Give a list of 10 mnemonic options based on these criteria. 
Give your output in JSON format.""",
        }

        # For some reason using tokenizer.apply_chat_template() here causes weird output
        input = self.messages + [final_message]
        print(input)
        output = self.pipe(input, **self.generation_args)
        response = output[0]["generated_text"]
        logger.debug(f"Generated cue: {response}")
        return response


if __name__ == "__main__":
    mnemonic_gen = MnemonicGen()
    mnemonic = mnemonic_gen.generate_mnemonic()
