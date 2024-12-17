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

        if len(lang_code) < 3:
            logger.error(
                f"Language code '{lang_code}' is too short. Please provide the ISO 639-3 language code (https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)."
            )
            return

        # Filter rows where Iso Code starts with the input
        language_rows = supported_languages[
            supported_languages["Iso Code"].str.startswith(lang_code)
        ]

        if language_rows.empty:
            logger.error(
                f"Language code '{lang_code}' is not supported. Please check the supported languages (https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)."
            )
            return

        self.lang_code = language_rows.iloc[0]["Iso Code"]

        # TODO Handle multiple languages
        if len(language_rows) > 1:
            logger.warning(
                f"Multiple TTS languages found for language code '{lang_code}'. We are now using {self.lang_code}, options are: {language_rows['Iso Code'].values}."
            )

        self.tokenizer = VitsTokenizer.from_pretrained(
            f"facebook/mms-tts-{self.lang_code}", cache_dir="models"
        )
        self.model = VitsModel.from_pretrained(
            f"facebook/mms-tts-{self.lang_code}", cache_dir="models"
        )

        self.pipe = pipeline(
            "text-to-speech", model=self.model, tokenizer=self.tokenizer
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


if __name__ == "__main__":
    tts = TTS("azj-script_latin")
    tts.tts("Sağ olun!")

    # To prevent VRAM usage kill the TTS instance after use
