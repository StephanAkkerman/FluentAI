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
        self.model_name = model_name if model_name else self.config.get("MODEL")
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
                "content": "You are a helpful assistant that generates mnemonics for foreign language vocabulary words. The mnemonic should sound very similar to the foreign word.  Use common English words or short phrases.",
            }
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
        if config.get("LLM").get("QUANTIZATION") == "4bit":
            logger.debug("Using 4-bit quantization for LLM")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif config.get("LLM").get("QUANTIZATION") == "8bit":
            logger.debug("Using 8-bit quantization for LLM")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda" if self.offload else "auto",
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir="models",
            # quantization_config=bnb_config,
        )

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
                "content": f"""Think of a mnemonic to remember the {language} word {word}.
            Think of an English word or a combination of 2 words that sound similar to how {word} would be pronounced in {language}.
            Also consider that the mnemonic should be an easy to imagine word and a word that is commonly used.
            Do not simply translate the word, the mnemonic should be a *memory aid* based on sound, not a translation.
            Give a list of 10 mnemonic options based on these criteria.
            Give your output in a Python list format. Like so ["option1", ..., "option10"]""",
            }

            output = self.pipe(
                self.mnemonic_messages + [final_message], **self.generation_args
            )
            response = output[0]["generated_text"]
            logger.debug(f"Generated mnemonics: {response[-1]['content']}")

            # Convert the string of a list to an actual list
            try:
                top = eval(response[-1]["content"])
            except Exception as e:
                # Retry generating using the LLM for 2 more times
                logger.error(f"Error parsing response: {e}")
                logger.debug("Retrying...")
                # TODO retry logic

        verbal_cue = self.generate_cue(
            word1=translated_word,
            word2=keyword if keyword else top[0],
        )

        return top, translated_word, transliterated_word, ipa, verbal_cue

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
    print(asyncio.run(vc.generate_mnemonic(word="flasche", language_code="ger")))
    # print(vc.generate_cue("bottle", "flashy"))
