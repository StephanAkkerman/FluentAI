import sys
from pathlib import Path

from git import GitCommandError, RemoteProgress, Repo
from tqdm import tqdm


def check_directory_exists(directory_path):
    """
    Checks if a directory exists at the specified path.

    Args:
        directory_path (str or Path): The path to the directory.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return Path(directory_path).is_dir()


def clone_repository(repo_url, clone_path):
    """
    Clones a GitHub repository to the specified path with a progress bar.

    Args:
        repo_url (str): The HTTPS or SSH URL of the GitHub repository.
        clone_path (str or Path): The local path where the repository will be cloned.
    """
    try:
        print(f"Cloning repository from {repo_url} to {clone_path}...")
        # Initialize CloneProgress with descriptive parameters
        Repo.clone_from(repo_url, clone_path, progress=CloneProgress())
        print("Repository cloned successfully.")
    except GitCommandError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)


class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=""):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


def get_clts():
    # Configuration
    data_directory = Path("data")  # Change this to your desired data directory
    repo_url = (
        "https://github.com/cldf-clts/clts.git"  # Replace with your repository URL
    )
    repo_name = (
        repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    )  # Extract repository name

    # Define the full path where the repository will be cloned
    clone_path = data_directory / repo_name

    # Check if the directory already exists
    if check_directory_exists(clone_path):
        print(f"The directory '{clone_path}' already exists. Skipping clone.")
    else:
        # Ensure the /data directory exists
        if not data_directory.exists():
            try:
                data_directory.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {data_directory}")
            except Exception as e:
                print(f"Failed to create directory '{data_directory}': {e}")
                sys.exit(1)

        # Clone the repository
        clone_repository(repo_url, clone_path)


if __name__ == "__main__":
    get_clts()