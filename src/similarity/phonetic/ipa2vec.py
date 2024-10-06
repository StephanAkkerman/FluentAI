import numpy as np
import panphon
from pyclts import CLTS
from soundvectors import SoundVectors

# Load CLTS data and SoundVectors
bipa = CLTS("data/clts-2.3.0").bipa
sv = SoundVectors(ts=bipa)


ft = panphon.FeatureTable()


def panphon_vec(ipa: str) -> list:
    return ft.word_to_vector_list(ipa, numeric=True)


def soundvec(ipa: str) -> list:
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
        print("Panphon", panphon_vec(input))
        print("Soundvec", soundvec(input))
