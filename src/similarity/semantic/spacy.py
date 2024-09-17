import spacy

# Download and load the spaCy model with vectors
nlp = spacy.load("en_core_web_md")  # or "en_core_web_lg"


def find_similar_spacy(word, n=5):
    token = nlp(word)
    if not token.has_vector:
        print(f"The word '{word}' does not have a vector representation.")
        return []
    sims = []
    for other_word in nlp.vocab:
        if other_word.has_vector and other_word.is_lower and other_word.is_alpha:
            sim = token.similarity(other_word)
            sims.append((other_word.text, sim))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims[:n]
