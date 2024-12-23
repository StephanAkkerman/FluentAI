import pandas as pd


def _map_iso_codes():
    # Define the custom mappings
    custom_mappings = {
        "East Armenian": ["hyw"],
        "West Armenian": ["hyw"],
        "Azerbaijani": ["azj-script_cyrillic", "azj-script_latin"],
        "Bashkir": ["bak"],
        "Min Nan": ["nan"],
        "British English": ["eng"],
        "American English": ["eng"],
        "French (Quebec)": ["fra"],
        "German": ["deu"],
        "Ancient Greek": ["grc"],
        "Jamaican Creole": ["jam"],
        "Kurdish": ["kmr-script_latin", "kmr-script_arabic", "kmr-script_cyrillic"],
        "Latin Classical": ["lat"],
        "Latin Ecclesiastical": ["lat"],
        "Northeastern Thai": ["tha"],
        "Papiamento": ["pap"],
        "Portuguese (Brazil)": ["por"],
        "Portuguese (Portugal)": ["por"],
        "Spanish (Latin America)": ["spa"],  # Added missing closing parenthesis
        "Spanish (Mexico)": ["spa"],
        "Vietnamese (Northern)": ["vie"],
        "Vietnamese (Southern)": ["vie"],
        "Vietnamese (Central)": ["vie"],
        "Welsh (North)": ["cym"],
        "Welsh (South)": ["cym"],
    }

    # Function to normalize language names
    def normalize_language(name):
        return (
            name.strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
            .replace(" (", " ")
            .replace(")", "")
        )

    # Normalize the keys in custom_mappings
    normalized_custom_mappings = {
        normalize_language(k): v for k, v in custom_mappings.items()
    }

    # Load JSON file
    json_file_path = "data/languages.json"
    # Read JSON as a Series (Assuming JSON structure: {"Language Name": "Iso Code", ...})
    json_series = pd.read_json(json_file_path, typ="series")

    # Convert Series to DataFrame
    json_df = json_series.to_frame().reset_index()
    json_df.columns = ["Language Name", "Iso Code_json"]

    # Load Parquet file
    parquet_file_path = "data/tts-languages.parquet"
    parquet_df = pd.read_parquet(parquet_file_path)

    print("JSON DataFrame:")
    print(json_df.head())

    print("\nParquet DataFrame:")
    print(parquet_df.head())

    # Apply normalization
    json_df["language_normalized"] = json_df["Language Name"].apply(normalize_language)
    parquet_df["language_normalized"] = parquet_df["Language Name"].apply(
        normalize_language
    )

    # Group parquet data by normalized language and aggregate ISO codes into lists
    parquet_grouped = (
        parquet_df.groupby("language_normalized")["Iso Code"].apply(list).reset_index()
    )
    parquet_grouped = parquet_grouped.rename(columns={"Iso Code": "Iso Code_parquet"})

    print("\nParquet Grouped DataFrame:")
    print(parquet_grouped.head())

    # Merge JSON and Parquet DataFrames on normalized language names
    merged_df = pd.merge(
        json_df,
        parquet_grouped,
        on="language_normalized",
        how="left",
        suffixes=("_json", "_parquet"),
    )

    print("\nMerged DataFrame (Before Applying Custom Mappings):")
    print(merged_df.head())

    # Apply custom mappings: overwrite Iso Code_parquet where applicable
    def apply_custom_mapping(row):
        if row["language_normalized"] in normalized_custom_mappings:
            return normalized_custom_mappings[row["language_normalized"]]
        else:
            return row["Iso Code_parquet"]

    merged_df["Iso Code_parquet"] = merged_df.apply(apply_custom_mapping, axis=1)

    print("\nMerged DataFrame (After Applying Custom Mappings):")
    print(merged_df.head())

    print(
        merged_df[["language_normalized", "Iso Code_json", "Iso Code_parquet"]].head(20)
    )

    # Identify unmatched languages (languages present in JSON but not in Parquet)
    unmatched = merged_df[merged_df["Iso Code_parquet"].isna()]
    print("\nUnmatched Languages:")
    print(unmatched["language_normalized"])

    # Save the merged DataFrame to a CSV file
    merged_df.to_parquet("data/g2p-to-tss-mapping.parquet", index=False)
    print("\nMapping saved to 'data/'")


if __name__ == "__main__":
    _map_iso_codes()
