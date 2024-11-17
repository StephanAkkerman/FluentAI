import os

import pandas as pd
from huggingface_hub import hf_hub_download

from fluentai.services.card_gen.utils.logger import logger


class VocabularyManager:
    def __init__(
        self,
        lang_code: str,
    ):
        """
        Initialize the VocabularyManager.

        Parameters
        ----------
        lang_code : str
            The language code for the vocabulary.
        last_word_file : str, optional
            Path to the file storing the last word, by default "last_word.txt".
        cache_dir : str, optional
            Directory to cache downloaded datasets, by default "datasets".
        """
        self.lang_code = lang_code
        self.last_word_file = f"data/{lang_code}_last_word.txt"

        # Download and load the vocabulary
        self.vocab_df = self._download_vocab()
        self.vocab_list = self.vocab_df["word"].tolist()

        # Load the last used word
        self.last_word = self._load_last_word()

    def _download_vocab(self) -> pd.DataFrame:
        """
        Download the vocabulary for the specified language.

        Returns
        -------
        pd.DataFrame
            The vocabulary DataFrame with 'word' and 'frequency' columns.

        Raises
        ------
        RuntimeError
            If both 50k and full vocabulary downloads fail.
        """
        try:
            vocab_path = hf_hub_download(
                repo_id="StephanAkkerman/frequency-words-2018",
                filename=f"{self.lang_code}/{self.lang_code}_50k.txt",
                cache_dir="datasets",
                repo_type="dataset",
            )
        except Exception:
            try:
                vocab_path = hf_hub_download(
                    repo_id="StephanAkkerman/frequency-words-2018",
                    filename=f"{self.lang_code}/{self.lang_code}_full.txt",
                    cache_dir="datasets",
                    repo_type="dataset",
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download vocabulary files for language '{self.lang_code}'."
                ) from e

        vocab = pd.read_csv(
            vocab_path,
            header=None,
            sep=" ",
            names=["word", "frequency"],
            encoding="utf-8",
        )

        if vocab.empty:
            raise ValueError(
                f"The downloaded vocabulary for '{self.lang_code}' is empty."
            )

        return vocab

    def _load_last_word(self) -> str:
        """
        Load the last used word from the tracking file.

        Returns
        -------
        str
            The last used word, or None if the file does not exist.
        """
        if os.path.exists(self.last_word_file):
            with open(self.last_word_file, encoding="utf-8") as file:
                last_word = file.read().strip()
            logger.info(f"Loaded last word from '{self.last_word_file}': '{last_word}'")
            return last_word
        else:
            logger.warn(
                f"No tracking file found at '{self.last_word_file}'. Starting from the first word."
            )
            return None

    def get_next_word(self) -> str:
        """
        Get the next word in the vocabulary after the last used word.

        Returns
        -------
        str
            The next word in the vocabulary, or None if at the end of the vocabulary.
        """
        next_word = None

        if self.last_word and self.last_word in self.vocab_list:
            last_index = self.vocab_list.index(self.last_word)
            next_index = last_index + 1
            if next_index < len(self.vocab_list):
                next_word = self.vocab_list[next_index]
            else:
                logger.error(
                    "Reached the end of the vocabulary. No next word available."
                )
        else:
            if self.last_word:
                logger.error(
                    f"Last word '{self.last_word}' not found in vocabulary. Starting from the first word."
                )
            next_word = self.vocab_list[0] if self.vocab_list else None
            if next_word is None:
                logger.error("Vocabulary is empty. No next word available.")

        if next_word:
            self._update_last_word(next_word)

        return next_word

    def _update_last_word(self, word: str):
        """
        Update the tracking file with the new last word.

        Parameters
        ----------
        word : str
            The new word to write to the tracking file.
        """
        with open(self.last_word_file, "w", encoding="utf-8") as file:
            file.write(word)
        self.last_word = word

    def get_current_word(self) -> str:
        """
        Get the current last used word.

        Returns
        -------
        str
            The current last used word, or None if no word has been used yet.
        """
        return self.last_word


# Example Usage
if __name__ == "__main__":
    # Initialize the VocabularyManager for Afrikaans ('af')
    vocab_manager = VocabularyManager(lang_code="af")

    print(f"The current word is: {vocab_manager.get_current_word()}")
    print(f"The next word is: {vocab_manager.get_next_word()}")
