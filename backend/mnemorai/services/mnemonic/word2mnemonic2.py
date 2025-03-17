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
            },
            {"role": "user", "content": prompt},
        ]

    def generate_mnemonic(self, word):
        mnemonic = self.model.generate(word)
        return mnemonic


if __name__ == "__main__":
    mnemonic_gen = MnemonicGen()
    mnemonic = mnemonic_gen.generate_mnemonic("apple")
    print(mnemonic)
