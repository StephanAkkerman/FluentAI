import pandas as pd
from huggingface_hub import hf_hub_download

from fluentai.constants.config import config
from fluentai.services.mnemonic.phonetic.utils.cache import load_from_cache

# Combine the IPA dataset with the frequency and imageability dataset
ipa = load_from_cache()

# Rename token_ort to word
ipa = ipa.rename(columns={"token_ort": "word"})

imageability = pd.read_csv(
    hf_hub_download(
        repo_id=config.get("IMAGEABILITY").get("PREDICTIONS").get("REPO"),
        filename=config.get("IMAGEABILITY").get("PREDICTIONS").get("FILE"),
        cache_dir="datasets",
        repo_type="dataset",
    )
)

frequency = pd.read_csv(
    hf_hub_download(
        repo_id="StephanAkkerman/English-Age-of-Acquisition",
        filename="en.aoa.csv",
        cache_dir="datasets",
        repo_type="dataset",
    )
)

# Rename Word to word
frequency = frequency.rename(columns={"Word": "word"})
# Keep AoA_Kup_lem and Freq_pm as the columns
frequency = frequency[["word", "AoA_Kup_lem", "Freq_pm"]]

ipa = ipa.merge(imageability, on="word", how="left")
ipa = ipa.merge(frequency, on="word", how="left")

# Rename word to token_ort
ipa = ipa.rename(columns={"word": "token_ort"})

# Fill NaNs for AoA_Kup_lem and Freq_pm and rename
ipa["aoa"] = ipa["AoA_Kup_lem"].fillna(15)
ipa["freq"] = ipa["Freq_pm"].fillna(0.1)

ipa.to_csv("datasets/ipa.csv", index=False)

print(ipa.head())
