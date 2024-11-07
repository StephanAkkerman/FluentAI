import gzip
import os
import shutil
import threading
import time
from functools import wraps

import requests
from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_vectors
from tqdm import tqdm

from fluentai.utils.logger import logger


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
        logger.info("ERROR: Something went wrong during the download.")
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
    logger.info(f"Extraction complete: {extracted_path}")


def download_fasttext(file_name: str = "cc.en.300.bin.gz"):
    """
    Download the specified FastText embedding model from the official website.

    Parameters
    ----------
    file_name : str, optional
        The name of the file to download, by default "cc.en.300.bin.gz"
    """
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
    logger.info(f"Starting download from {url}")
    download_file(url, gz_path)
    logger.info(f"Downloaded file saved to {gz_path}")

    # Extract the .gz file to obtain the .bin file
    logger.info(f"Starting extraction of {gz_path}")
    extract_gz(gz_path, bin_path)

    # Delete the original .gz file to save space
    try:
        os.remove(gz_path)
        logger.info(f"Deleted the compressed file: {gz_path}")
    except OSError as e:
        logger.info(f"Error deleting file {gz_path}: {e}")

    logger.info("All operations completed successfully.")


def loading_indicator_with_estimated_time(desc="Loading...", estimated_time=50):
    """
    Decorator to display a loading progress bar with estimated time while a blocking function executes.

    Args:
        desc (str): Description to display alongside the progress bar.
        estimated_time (int or float): Estimated time in seconds for the function to execute.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Containers for function result and exceptions
            result_container = []
            exception_container = []
            done_event = threading.Event()

            def target():
                try:
                    result = func(*args, **kwargs)
                    result_container.append(result)
                except Exception as e:
                    exception_container.append(e)
                finally:
                    done_event.set()

            # Start the target function in a separate thread
            thread = threading.Thread(target=target)
            thread.start()

            # Initialize tqdm progress bar
            with tqdm(
                total=estimated_time,
                bar_format="{l_bar}{bar} | {elapsed}/{total} seconds",
                desc=desc,
                ncols=70,
                leave=True,  # Keep the progress bar after completion
            ) as pbar:
                start_time = time.time()
                while not done_event.is_set():
                    elapsed = time.time() - start_time
                    pbar.n = min(elapsed, estimated_time)  # Prevent overflow
                    pbar.refresh()

                    if elapsed >= estimated_time:
                        # Update description to indicate delay
                        pbar.set_description(f"{desc} (taking longer than estimated)")

                    time.sleep(0.1)  # Update interval

                # Once the function is done, set progress to 100%
                pbar.n = pbar.total
                pbar.refresh()
                pbar.set_description(f"{desc} - Completed")

            # Wait for the thread to finish
            thread.join()

            # Raise exception if occurred in the target function
            if exception_container:
                raise exception_container[0]

            # Return the function's result
            return result_container[0]

        return wrapper

    return decorator


@loading_indicator_with_estimated_time(desc="Loading FastText Model", estimated_time=50)
def _get_fasttext_from_keyedvectors(embedding_model_path="models/cc.en.300.model"):
    return FastTextKeyedVectors.load(embedding_model_path)


def get_fasttext_model(
    model_name="cc.en.300.bin", embedding_model_path="models/cc.en.300.model"
) -> FastTextKeyedVectors:
    """
    Download the specified embedding model and save it locally.

    Another option is: "wiki-news-300d-1M-subword.bin" and "models/wiki-news-300d-1M-subword.model"

    Args:
        model_name (str): Name of the model to download.
        embedding_model_path (str): Path to save the downloaded model.
    """
    # Check if the model already exists
    if os.path.exists(embedding_model_path):
        return _get_fasttext_from_keyedvectors(embedding_model_path)

    # Check if the .bin file already exists
    if not os.path.exists(f"data/fasttext_embeddings/{model_name}"):
        # Download the model .bin file with .gz extension
        download_fasttext(f"{model_name}.gz")

    # Load the model from the .bin file
    logger.info("Loading FastText embeddings...")
    embedding_model = load_facebook_vectors(f"data/fasttext_embeddings/{model_name}")
    embedding_model.save(embedding_model_path)
    logger.info(f"Model saved locally at '{embedding_model_path}'.")

    return embedding_model


fasttext_model = get_fasttext_model()
