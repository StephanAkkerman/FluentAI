import gensim.downloader as api


def load_model():
    print("Loading GloVe model. This may take a while...")
    model = api.load("glove-wiki-gigaword-100")  # You can choose other models
    print("Model loaded successfully.")
    return model


def find_top_similar_words(word, model, topn=5):
    try:
        similar_words = model.most_similar(word, topn=topn)
        return similar_words
    except KeyError:
        print(f"The word '{word}' is not in the vocabulary.")
        return []


def main():
    model = load_model()
    while True:
        word = (
            input("\nEnter an English word (or type 'exit' to quit): ").strip().lower()
        )
        if word == "exit":
            print("Exiting the program.")
            break
        similar = find_top_similar_words(word, model)
        if similar:
            print(f"\nTop {len(similar)} words similar to '{word}':")
            for idx, (sim_word, score) in enumerate(similar, start=1):
                print(f"{idx}. {sim_word} (Similarity: {score:.4f})")


if __name__ == "__main__":
    main()
