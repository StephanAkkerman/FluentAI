import gzip
import os
import shutil

import requests
from tqdm import tqdm


def download_file(url, dest_path, chunk_size=1024):
    """
    Downloads a file from a URL to a specified destination path with a progress bar.

    Args:
        url (str): The URL of the file to download.
        dest_path (str): The full path (including filename) where the file will be saved.
        chunk_size (int, optional): The size of each chunk to read during download. Defaults to 1024.
    """
    # Send a HTTP GET request with stream=True
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Get the total file size from headers
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = chunk_size  # 1 Kilobyte
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading"
    )

    with open(dest_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Something went wrong during the download.")
        raise Exception("Download incomplete.")


def extract_gz(gz_path, extracted_path):
    """
    Extracts a .gz file to a specified location.

    Args:
        gz_path (str): The path to the .gz file.
        extracted_path (str): The path where the extracted file will be saved.
    """
    with gzip.open(gz_path, "rb") as f_in:
        with open(extracted_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extraction complete: {extracted_path}")


def download_fasttext(file_name: str = "cc.en.300.bin.gz"):
    # URL of the .gz file
    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{file_name}"

    # Define the directory where you want to save the files
    # You can change this to your desired directory
    download_directory = "data/fasttext_embeddings"
    os.makedirs(
        download_directory, exist_ok=True
    )  # Create directory if it doesn't exist

    # Define the paths for the downloaded .gz file and the extracted .bin file
    gz_path = os.path.join(download_directory, file_name)
    bin_path = os.path.join(download_directory, file_name.split(".gz")[0])

    # Download the .gz file
    print(f"Starting download from {url}")
    download_file(url, gz_path)
    print(f"Downloaded file saved to {gz_path}")

    # Extract the .gz file to obtain the .bin file
    print(f"Starting extraction of {gz_path}")
    extract_gz(gz_path, bin_path)

    # Delete the original .gz file to save space
    try:
        os.remove(gz_path)
        print(f"Deleted the compressed file: {gz_path}")
    except OSError as e:
        print(f"Error deleting file {gz_path}: {e}")

    print("All operations completed successfully.")


if __name__ == "__main__":
    download_fasttext()
