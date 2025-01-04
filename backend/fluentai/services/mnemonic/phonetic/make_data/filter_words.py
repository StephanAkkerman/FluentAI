import nltk
import pandas as pd
from huggingface_hub import hf_hub_download
from nltk import pos_tag
from nltk.corpus import wordnet, words
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from fluentai.constants.config import config

# Download required NLTK data
nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt")
nltk.download("wordnet")


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and lowercase the words corpus
nltk_words = set(word.lower() for word in words.words())

# Get the words list from huggingface
eng_ipa = pd.read_csv(
    hf_hub_download(
        repo_id=config.get("PHONETIC_SIM").get("IPA").get("REPO"),
        filename=config.get("PHONETIC_SIM").get("IPA").get("FILE"),
        cache_dir="datasets",
        repo_type="dataset",
    )
)

# Get the words
your_words = eng_ipa["token_ort"].tolist()

# Drop any non string values
your_words = [word for word in your_words if isinstance(word, str)]


# Function to lemmatize words
def _lemmatize_word(word):
    lemma = lemmatizer.lemmatize(word, pos="v")
    if lemma != word:
        return lemma
    lemma = lemmatizer.lemmatize(word, pos="n")
    return lemma


# Function to check if word is a common noun, verb, or adjective
def _is_common_word(word):
    pos = pos_tag([word])[0][1]
    return pos.startswith("NN") or pos.startswith("VB") or pos.startswith("JJ")


def _is_in_wordnet(word):
    return bool(wordnet.synsets(word))


# Filter common words
common_words = []
dropped_words = []
for word in tqdm(your_words):
    lower_word = word.lower()
    lemma = _lemmatize_word(lower_word)
    if lemma in nltk_words and _is_common_word(word) and _is_in_wordnet(lemma):
        common_words.append(word)

# Use the common words to filter the dataset
filtered_eng_ipa = eng_ipa[eng_ipa["token_ort"].isin(common_words)]

# Save the filtered dataset
file_name = (
    config.get("PHONETIC_SIM").get("IPA").get("FILE").split(".")[0] + "_filtered"
)
filtered_eng_ipa.to_csv(f"data/{file_name}.csv", index=False)
