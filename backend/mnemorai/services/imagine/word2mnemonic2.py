import asyncio

import pandas as pd
from backend.mnemorai.services.imagine.grapheme2phoneme import Grapheme2Phoneme
from backend.mnemorai.services.imagine.translator import translate_word
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from mnemorai.constants.config import config
from mnemorai.constants.languages import G2P_LANGUAGES
from mnemorai.logger import logger


class MnemonicGen:
    def __init__(self):
        self.config = config.get("LLM")
        self.g2p_model = Grapheme2Phoneme()

        # https://docs.unsloth.ai/get-started/all-our-models
        self.model_name = (
            # "microsoft/Phi-3-mini-4k-instruct" #only works with transformers 4.47.1 or older
            # "Qwen/Qwen2.5-7B-Instruct"
            # "Qwen/Qwen2.5-14B-Instruct"
            # "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
            # "Qwen/Qwen2.5-14B-Instruct-GGUF"
            # "microsoft/Phi-4-mini-instruct"
        )

        # use recommended settings for phi-3.5
        self.generation_args = {
            "max_new_tokens": 512,
            # "do_sample": False,  # Default = False
            # "num_beams": 1,  # Default = 1
            # "top_k": 50,
            # # "top_p": 0.95,
            # "return_full_text": False,
            # "temperature": 0.6,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir="models",
            # gguf_file=gguf,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="models",
            # gguf_file=gguf,
        )
        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype="float16",
        )

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates mnemonics for foreign language vocabulary words. The mnemonic should sound very similar to the foreign word.  Use common English words or short phrases.",
            }
        ]

    def word2ipa(
        self,
        word: str,
        language_code: str,
    ) -> str:
        """
        Get the IPA representation of a word.

        Parameters
        ----------
        word : str
            The word to convert to IPA
        language_code : str, optional
            The language code of the word, by default "eng-us"

        Returns
        -------
        str
            The IPA representation of the word
        """
        # Try searching in the dataset
        if "eng-us" in language_code:
            # First try lookup in the .tsv file
            logger.debug("Loading the IPA dataset")
            eng_ipa = pd.read_csv(
                hf_hub_download(
                    repo_id=config.get("PHONETIC_SIM").get("IPA").get("REPO"),
                    filename=config.get("PHONETIC_SIM").get("IPA").get("FILE"),
                    cache_dir="datasets",
                    repo_type="dataset",
                )
            )

            # Check if the word is in the dataset
            ipa = eng_ipa[eng_ipa["token_ort"] == word]["token_ipa"]

            if not ipa.empty:
                return ipa.values[0].replace(" ", "")

        # Use the g2p models
        return self.g2p_model.g2p([f"<{language_code}>:{word}"])

    async def generate_mnemonic(
        self,
        word: str,
        language_code: str,
        keyword: str = None,
        key_sentence: str = None,
    ) -> tuple:
        """
        Generate a mnemonic for the input word using the phonetic representation.

        Parameters
        ----------
        word : str
            Foreign word to generate mnemonic for.
        language_code : str
            Language code of the input word.
        keyword : str, optional
            User-provided keyword to use in the mnemonic.
        key_sentence : str, optional
            User-provided key sentence to use as the mnemonic.

        Returns
        -------
        tuple
            A tuple containing the top matches, translated word, transliterated word, and IPA.
        """
        # Convert the input word to IPA representation
        ipa = self.phonetic_sim.word2ipa(word=word, language_code=language_code)

        translated_word, transliterated_word = await translate_word(word, language_code)

        # If a keyword or key sentence is provided, use it directly for scoring
        if keyword or key_sentence:
            return None, translated_word, transliterated_word, ipa

        language = G2P_LANGUAGES.get(language_code, "Unknown Language")
        final_message = {
            "role": "user",
            "content": f"""Think of a mnemonic to remember the {language} word {word}.
        Think of an English word or a combination of 2 words that sound similar to how {word} would be pronounced in {language}.
        Also consider that the mnemonic should be an easy to imagine word and a word that is commonly used.
        Do not simply translate the word, the mnemonic should be a *memory aid* based on sound, not a translation.
        Give a list of 10 mnemonic options based on these criteria.
        Give your output in a Python list format. Like so ["option1", ..., "option10"]""",
        }

        # For some reason using tokenizer.apply_chat_template() here causes weird output
        input = self.messages + [final_message]
        logger.debug(input)
        output = self.pipe(input, **self.generation_args)
        response = output[0]["generated_text"]
        logger.debug(f"Generated cue: {response}")

        # Convert the string of a list to an actual list
        try:
            top = eval(response[-1]["content"])
        except Exception as e:
            # Retry generating using the LLM for 2 more times
            logger.error(f"Error parsing response: {e}")
            logger.debug("Retrying...")
            # TODO retry logic
        return top, translated_word, transliterated_word, ipa


if __name__ == "__main__":
    mnemonic_gen = MnemonicGen()
    print(
        asyncio.run(mnemonic_gen.generate_mnemonic(word="daging", language_code="ind"))
    )
