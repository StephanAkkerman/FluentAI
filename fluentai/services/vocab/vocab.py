import pandas as pd
from huggingface_hub import hf_hub_download


def download_vocab(lang_code: str):
    """
    Download the vocabulary for a specific language.

    Parameters
    ----------
    lang_code : str
        The language code for the vocabulary.

    Returns
    -------
    pd.DataFrame
        The vocabulary for the specified language.
    """
    try:
        vocab = pd.read_csv(
            hf_hub_download(
                "StephanAkkerman/frequency-words-2018",
                f"{lang_code}/{lang_code}_50k.txt",
                cache_dir="datasets",
                repo_type="dataset",
            ),
            header=None,
            sep=" ",
            names=["word", "frequency"],
        )
    except Exception as e:
        # Try the full set
        vocab = pd.read_csv(
            hf_hub_download(
                "StephanAkkerman/frequency-words-2018",
                f"{lang_code}/{lang_code}_full.txt",
                cache_dir="datasets",
                repo_type="dataset",
            ),
            header=None,
            sep=" ",
            names=["word", "frequency"],
        )

    return vocab


def get_vocab(lang_code: str):
    """
    Get the vocabulary for a specific language.

    Parameters
    ----------
    lang_code : str
        The language code for the vocabulary.

    Returns
    -------
    pd.DataFrame
        The vocabulary for the specified language.
    """
    # Find the last vocab word that was downloaded
    last_vocab_word = None

    # Load the vocabulary from the file
    vocab_df = download_vocab(lang_code)

    # Use the vocab after the last word

    return vocab_df


if __name__ == "__main__":
    print(get_vocab("af").head())
