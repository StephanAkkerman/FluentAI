import os

import pandas as pd
import scipy
from transformers import VitsModel, VitsTokenizer, pipeline

from fluentai.services.card_gen.utils.logger import logger

# Check if the language code is supported
supported_languages = pd.read_parquet("data/tts-languages.parquet")


def _generate_lang_codes():
    """Generates the language codes for the TTS service."""
    import requests
    from bs4 import BeautifulSoup

    html_content = requests.get(
        "https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html"
    )
    # Override with utf-8
    html_content.encoding = "utf-8"

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content.text, "html.parser")

    # Find all <p> elements
    p_elements = soup.find_all("p")[1:]  # Skip the first <p> (header)

    # Extract Iso Code and Language Name
    data = []
    for p in p_elements:
        text = p.get_text(strip=True)
        iso_code, language_name = text.split("   ")
        data.append([iso_code.strip(), language_name.strip()])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Iso Code", "Language Name"])

    # Display the DataFrame
    df.to_parquet("data/tts-languages.parquet", index=False)


class TTS:
    def __init__(self, lang_code: str):
        os.makedirs("local_data", exist_ok=True)
        os.makedirs("local_data/tts", exist_ok=True)

        self.lang_code = get_mapping(lang_code)

        # TODO Handle multiple languages
        if len(self.lang_code) > 1:
            logger.warning(
                f"Multiple TTS languages found for language code '{lang_code}'. We are now using {self.lang_code[0]}, options are: {self.lang_code}."
            )
        self.lang_code = self.lang_code[0]

        self.tokenizer = VitsTokenizer.from_pretrained(
            f"facebook/mms-tts-{self.lang_code}", cache_dir="models"
        )
        self.model = VitsModel.from_pretrained(
            f"facebook/mms-tts-{self.lang_code}", cache_dir="models"
        )

        self.pipe = pipeline(
            "text-to-speech", model=self.model, tokenizer=self.tokenizer, device="cpu"
        )

    def tts(self, text: str, file_name: str = "tts") -> str:
        """Generate a TTS audio file from the input text.

        The generated audio file will be saved in the local_data/tts directory.
        This filename can be removed after adding it to a card.

        Note: If there are ever issues with the input text
        Try using uroman to convert the text to latin script
        https://github.com/isi-nlp/uroman

        Parameters
        ----------
        text : str
            The input text to convert to speech

        Returns
        -------
        str
            The path to the generated audio file
        """
        try:
            out = self.pipe(text)
            audio = out.get("audio")
            sampling_rate = out.get("sampling_rate")
        except Exception as e:
            logger.error(
                f"Failed to generate TTS audio: {e}. This could be due to a mismatch in the language code."
            )
            return

        # Save it to a file
        try:
            scipy.io.wavfile.write(
                f"local_data/tts/{file_name}.wav",
                rate=sampling_rate,
                data=audio[0],
            )
        except Exception as e:
            logger.error(f"Failed to save TTS audio: {e}")
            return

        return f"local_data/tts/{file_name}.wav"


def get_mapping(lang_code: str) -> list:
    """Given the G2P language code converts it to the TTS language code.

    Parameters
    ----------
    lang_code : str, optional
        The G2P language code

    Returns
    -------
    list
        The TTS language code(s).
    """
    mappings = pd.read_parquet("data/g2p-to-tss-mapping.parquet")
    row = mappings[mappings["Iso Code_json"].str.contains(lang_code)]
    if row.empty:
        return []
    else:
        return row["Iso Code_parquet"].values[0]


if __name__ == "__main__":
    tts = TTS("dut")
    tts.tts("Hallo allemaal, ik spreek Nederlands!")
