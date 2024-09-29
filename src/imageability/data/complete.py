from datasets import load_dataset

imageability_corpus = load_dataset("StephanAkkerman/imageability-corpus")

# Convert to a pandas DataFrame
imageability_corpus_df = imageability_corpus.to_pandas()

# https://websites.psychology.uwa.edu.au/school/mrcdatabase/uwa_mrc.htm
# https://github.com/samzhang111/mrc-psycholinguistics/blob/master/wordmodel.py
mrc = load_dataset("StephanAkkerman/MRC-psycholinguistic-database")
mrc_df = mrc.to_pandas()

# Change word column to lowercase
mrc_df["word"] = mrc_df["Word"].str.lower()
