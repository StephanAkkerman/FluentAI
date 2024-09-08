import numpy as np
import panphon
from pyclts import CLTS
from soundvectors import SoundVectors

# Load CLTS data and SoundVectors
bipa = CLTS("data/clts-2.3.0").bipa
sv = SoundVectors(ts=bipa)


ft = panphon.FeatureTable()


def panphon_vec(word) -> list:
    return ft.word_to_vector_list(word, numeric=True)


def soundvec(word) -> list:
    word_vector = []
    for letter in word:
        try:
            word_vector.append(sv.get_vec(letter))
        except ValueError:
            word_vector.append(np.zeros(len(sv.get_vec("a"))))  # Handle unknown letters
    return word_vector
