import itertools
import sys

sys.path.append(".")
import numpy as np
import panphon
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset
from src.g2p import g2p

ft = panphon.FeatureTable()


def load_data():
    # Load the dataset
    ds = load_dataset("zouharvi/pwesuite-eval", cache_dir="datasets", split="train")

    # Filter for English language entries
    ds = ds.filter(lambda x: x["lang"] == "en")

    # Extract all phonemes from IPA and ARPAbet transcriptions
    ipa_phonemes = set(itertools.chain.from_iterable(ds["token_ipa"]))

    # Create a mapping from phoneme to index
    phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(ipa_phonemes)}

    return ds, phoneme_to_index


def one_hot_encode_phonemes(transcription, phoneme_to_index):
    one_hot_vectors = []
    for phoneme in transcription:
        one_hot_vector = np.zeros(len(phoneme_to_index))
        one_hot_vector[phoneme_to_index[phoneme]] = 1
        one_hot_vectors.append(one_hot_vector)
    return np.array(one_hot_vectors)


def concatenate_vectors(one_hot_vectors):
    return one_hot_vectors.flatten()


def average_pooling(one_hot_vectors):
    return np.mean(one_hot_vectors, axis=0)


# Choose a method to combine the vectors
def create_word_embedding(transcription, phoneme_to_index, method="average"):
    one_hot_vectors = one_hot_encode_phonemes(transcription, phoneme_to_index)
    if method == "concatenate":
        embedding = concatenate_vectors(one_hot_vectors)
    elif method == "average":
        embedding = average_pooling(one_hot_vectors)
    else:
        raise ValueError("Invalid method chosen. Use 'concatenate' or 'average'.")

    return embedding


# Function to process the dataset with vectorizer and PCA
def process_dataset_with_vectorizer(
    ds, transcription_type="ipa", vectorizer_type="tfidf", use_pca=True
):
    # Create a vectorizer
    vectorizer_args = {
        "max_features": 1024,
        "ngram_range": (1, 3),
        "stop_words": None,
        "analyzer": "char",
        "min_df": 1,
    }

    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_args)
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(**vectorizer_args)

    # Get the transcription list based on type
    if transcription_type == "ipa":
        transcriptions = [" ".join(ft.ipa_segs(x["token_ipa"])) for x in ds]
    else:
        transcriptions = [" ".join(x["token_ort"]) for x in ds]

    # Transform transcriptions using the vectorizer
    transformed_data = vectorizer.fit_transform(transcriptions)
    transformed_data = np.asarray(transformed_data.todense())

    # Apply PCA if required
    pca = None
    if use_pca:
        pca = PCA(n_components=300, whiten=True)
        transformed_data = pca.fit_transform(transformed_data)

    return vectorizer, transformed_data, pca


# Function to find top N similar sounding words
def get_top_n_similar_words(
    word,
    ds,
    phoneme_to_index,
    vectorizer,
    precomputed_embeddings,
    pca=None,
    transcription_type="ipa",
    n=5,
):
    # Get the transcription for the input word
    # transcription = ds.filter(lambda x: x["token_ort"] == word)[0][
    #     f"token_{transcription_type}"
    # ]
    transcription = word

    # Create the word embedding for the input word
    new_word_embedding = create_word_embedding(
        transcription, phoneme_to_index, method="average"
    ).reshape(1, -1)

    # Transform the new word embedding using the same vectorizer
    new_word_transformed = vectorizer.transform([" ".join(transcription)])
    new_word_transformed = np.asarray(new_word_transformed.todense())

    # Apply PCA if used
    if pca is not None:
        new_word_transformed = pca.transform(new_word_transformed)

    # Compute similarities
    similarities = cosine_similarity(
        new_word_transformed, precomputed_embeddings
    ).flatten()
    sorted_indices = np.argsort(-similarities)  # Sort in descending order

    top_n_indices = sorted_indices[1 : n + 1]  # Exclude the word itself
    top_n_words = [ds[int(i)]["token_ort"] for i in top_n_indices]

    return top_n_words


def get_IPA(word: str = "Kucing", language: str = "ind", phoneme_to_index: dict = None):
    ipa = g2p([f"<{language}>: {word}"])[0]
    # filter characters not in the phoneme_to_index
    ipa = "".join([c for c in ipa if c in phoneme_to_index])
    return ipa


def get_top_x(word: str = "kucing", language: str = "ind", n: int = 5):
    ds, phoneme_to_index = load_data()

    # Process the dataset
    vectorizer, precomputed_embeddings, pca = process_dataset_with_vectorizer(
        ds, transcription_type="ipa"
    )

    top_5_similar_words = get_top_n_similar_words(
        get_IPA("kucing", language, phoneme_to_index),
        ds,
        phoneme_to_index,
        vectorizer,
        precomputed_embeddings,
        pca=pca,
        transcription_type="ipa",
        n=n,
    )
    print(f"Top {n} similar sounding words to '{word}': {top_5_similar_words}")


# TODO: combination of 2 english words
get_top_x()
