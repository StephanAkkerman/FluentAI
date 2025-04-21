import ast
import asyncio

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from mnemorai.constants.config import config
from mnemorai.constants.languages import G2P_LANGUAGES
from mnemorai.logger import logger
from mnemorai.services.pre.grapheme2phoneme import Grapheme2Phoneme
from mnemorai.services.pre.translator import translate_word
from mnemorai.utils.model_mem import manage_memory


class VerbalCue:
    def __init__(self, model_name: str = None):
        self.config = config.get("LLM")
        self.offload = self.config.get("OFFLOAD")
        self.model_name = model_name if model_name else self.config.get("MEDIUM_MODEL")
        self.g2p_model = Grapheme2Phoneme()

        # Use 512 tokens as default for generation
        self.generation_args = {
            "max_new_tokens": 512,
        }
        # Add the config parameters
        config_params = self.config.get("PARAMS")
        if config_params:
            self.generation_args.update(config_params)

        logger.debug(f"LLM generation args: {self.generation_args}")

        self.mnemonic_messages = [
            {
                "role": "system",
                "content": """You are a mnemonic=generation assistant.  
                When given a foreign word, you must produce **exactly 10** English mnemonics that:

                1. **Sound very similar** (phonetic similarity)  
                2. **Look similar** (orthographic similarity)  
                3. Are **semantically related** or evoke a related image  
                4. Are **common, easy to imagine** words or combination of two words

                **Scoring:**  
                • phonetic_similarity - weight 0.4  
                • orthographic_similarity - weight 0.3  
                • semantic_relatedness - weight 0.2  
                • imageability - weight 0.1  

                Compute a single composite **score** in [0.00-1.00] for each mnemonic using those weights, then **sort descending** by score.

                **Do not** translate the word.  Mnemonics must be memory aids based on sound and imagery.

                **Output** a Python literal list of dicts, one per line, exactly like this (no extra keys):
                [
                    {'mnemonic': '...', 'score': 0.95},
                    {'mnemonic': '...', 'score': 0.90},
                    ...
                    {'mnemonic': '...', 'score': 0.50},
                ]""",
            },
            {
                "role": "user",
                "content": """Generate mnemonics to remember the German word 'flasche'.""",
            },
            {
                "role": "assistant",
                "content": """
                [
                    {'mnemonic': 'flashy',   'score': 0.91},
                    {'mnemonic': 'flash',    'score': 0.89},
                    {'mnemonic': 'flask',    'score': 0.87},
                    {'mnemonic': 'flasher',  'score': 0.85},
                    {'mnemonic': 'fleshy',   'score': 0.82},
                    {'mnemonic': 'flusher',  'score': 0.78},
                    {'mnemonic': 'flush',    'score': 0.76},
                    {'mnemonic': 'flash he', 'score': 0.65},
                    {'mnemonic': 'flesh he', 'score': 0.60},
                    {'mnemonic': 'flossier', 'score': 0.55},
                ]
                """,
            },
        ]

        self.verbal_cue_messages = [
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
        q = config.get("LLM", {}).get("QUANTIZATION")
        if q == "4bit":
            logger.debug("Using 4-bit quantization for LLM")
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        elif q == "8bit":
            logger.debug("Using 8-bit quantization for LLM")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # build up a dict of kwargs
        kwargs = {
            "device_map": "cuda" if self.offload else "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "cache_dir": "models",
        }
        # only include quantization_config if we actually have one
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        # Check if LoRA should be enabled
        if config.get("LLM").get("USE_LORA"):
            lora = config.get("LLM").get("LORA")
            logger.debug(f"Loading LoRA ({lora}) for LLM")
            # Load the model with LoRA
            self.model = PeftModel.from_pretrained(self.model, lora)
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
            A tuple containing the top matches, translated word, transliterated word, IPA, and verbal cue.
        """
        # Convert the input word to IPA representation
        ipa = self.g2p_model.word2ipa(word=word, language_code=language_code)

        # Translate and transliterate the word
        translated_word, transliterated_word = await translate_word(word, language_code)

        # If a keyword or key sentence is provided do not generate mnemonics
        if not (keyword or key_sentence):

            language = G2P_LANGUAGES.get(language_code, "Unknown Language")
            final_message = {
                "role": "user",
                "content": f"""Generate mnemonics to remember the {language} word '{word}'.""",
            }

            output = self.pipe(
                self.mnemonic_messages + [final_message], **self.generation_args
            )
            response = output[0]["generated_text"]
            logger.debug(f"Generated mnemonics: {response[-1]['content']}")

            # parse the string into Python objects
            options = ast.literal_eval(response[-1]["content"])

            # find the dict with the highest score
            best = max(options, key=lambda opt: opt["score"])

        verbal_cue = self.generate_cue(
            word1=translated_word,
            word2=keyword if keyword else best,
        )

        return best, translated_word, transliterated_word, ipa, verbal_cue

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
            "content": f"Generate a mnemonic sentence for the given input. Start the sentence with 'imagine' and keep it simple. \n Input: English Word: {word1} | Mnemonic Word: {word2}",
        }
        # For some reason using tokenizer.apply_chat_template() here causes weird output
        output = self.pipe(
            self.verbal_cue_messages + [final_message], **self.generation_args
        )
        response = output[0]["generated_text"]
        logger.debug(f"Generated verbal cue: {response}")

        return response


if __name__ == "__main__":
    vc = VerbalCue()
    print(asyncio.run(vc.generate_mnemonic(word="daging", language_code="ind")))
    # print(vc.generate_cue("bottle", "flashy"))
