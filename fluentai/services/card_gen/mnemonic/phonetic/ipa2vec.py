import os

import numpy as np
import panphon
from pyclts import CLTS
from soundvectors import SoundVectors

from fluentai.services.card_gen.mnemonic.phonetic.clts_utils import get_clts
from fluentai.services.card_gen.utils.logger import logger

# Test if data/clts exists
if not os.path.exists("local_dataclts"):
    get_clts()

# Load CLTS data and SoundVectors
bipa = CLTS("local_data/clts").bipa
sv = SoundVectors(ts=bipa)


ft = panphon.FeatureTable()


def panphon_vec(ipa: str) -> list:
    """
    Use the panphon library to convert an IPA string to a list of feature vectors.

    Parameters
    ----------
    ipa : str
        The IPA string to convert to feature vectors.

    Returns
    -------
    list
        The list of feature vectors.
    """
    return ft.word_to_vector_list(ipa, numeric=True)


def soundvec(ipa: str) -> list:
    """
    Transform an IPA string to a list of sound vectors using the SoundVectors library.

    Parameters
    ----------
    ipa : str
        The IPA string to convert to sound vectors.

    Returns
    -------
    list
        The list of sound vectors.
    """
    word_vector = []
    for letter in ipa:
        try:
            word_vector.append(sv.get_vec(letter))
        except ValueError:
            word_vector.append(np.zeros(len(sv.get_vec("a"))))  # Handle unknown letters
    return word_vector


if __name__ == "__main__":
    # Example usage: grumble
    ipa_input = "ˈɡɹəmbəɫ"  # G2P
    ipa_input2 = "ɡɹʌmbl̩"  # wiktionary
    for input in [ipa_input, ipa_input2]:
        logger.info("Panphon", panphon_vec(input))
        logger.info("Soundvec", soundvec(input))
