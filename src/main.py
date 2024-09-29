import pandas as pd

# Load your data
df = pd.read_csv("data/imageability/mrc2.csv")

# Rename the columns
df = df.rename(
    columns={
        "word": "Word",
        "nlet": "Number of Letters",
        "nphon": "Number of Phonemes",
        "nsyl": "Number of Syllables",
        "kf_freq": "KF Written Frequency",
        "kf_ncats": "KF Number of Categories",
        "kf_nsamp": "KF Number of Samples",
        "tl_freq": "Thorndike-Lorge Frequency",
        "brown_freq": "Brown Verbal Frequency",
        "fam": "Familiarity",
        "conc": "Concreteness",
        "imag": "Imageability",
        "meanc": "Meaningfulness: Coloradao Norms",
        "meanp": "Meaningfulness: Pavio Norms",
        "aoa": "Age of Acquisition Rating",
        "tq2": "Word Type",
        "wtype": "Comprehensive Syntactic Category",
        "pdwtype": "Common Part of Speech",
        "alphasyl": "Morphemic status",
        "status": "Contextual Status",
        "var": "Pronunciation Variability",
        "cap": "Capitalization",
        "irreg": "Irregular Plural",
        "phon": "Stress-Marked Phonetic Transcription",
        "dphon": "Syllabified Phonetic Transcription",
        "stress": "Stress Pattern",
    }
)

# Define the new column order
new_column_order = [
    "Word",
    "Number of Letters",
    "Number of Phonemes",
    "Number of Syllables",
    "KF Written Frequency",
    "KF Number of Categories",
    "KF Number of Samples",
    "Thorndike-Lorge Frequency",
    "Brown Verbal Frequency",
    "Familiarity",
    "Concreteness",
    "Imageability",
    "Meaningfulness: Coloradao Norms",
    "Meaningfulness: Pavio Norms",
    "Age of Acquisition Rating",
    "Word Type",
    "Comprehensive Syntactic Category",
    "Common Part of Speech",
    "Morphemic status",
    "Contextual Status",
    "Pronunciation Variability",
    "Capitalization",
    "Irregular Plural",
    "Stress-Marked Phonetic Transcription",
    "Syllabified Phonetic Transcription",
    "Stress Pattern",
]

# Reorder the DataFrame columns
df = df[new_column_order]
df.to_csv("data/imageability/mrc_psycholinguistic_database.csv", index=False)
