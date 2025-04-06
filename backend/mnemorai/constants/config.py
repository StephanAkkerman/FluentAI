import argparse
import os
from pathlib import Path

import yaml


def load_config_path() -> str:
    """
    Determine the path to the configuration file.

    The priority order is:
    1. `mnemorai_CONFIG_PATH` environment variable.
    2. Command-line argument `--config`.
    3. Default path: `Path(os.getcwd()).parent.parent.parent / "config.yaml"`.
       This is the path to the config.yaml file in the root directory of the repository.

    Returns
    -------
    str
        The path to the configuration file.

    Raises
    ------
    FileNotFoundError
        If no configuration file is found via any method.
    """
    # Check environment variable
    if "mnemorai_CONFIG_PATH" in os.environ:
        return os.environ["mnemorai_CONFIG_PATH"]

    # Command-line argument
    parser = argparse.ArgumentParser(description="Load configuration for mnemorai.")
    parser.add_argument("--config", help="Path to the config.yaml file")
    args = parser.parse_args()
    if args.config:
        return args.config

    # Default path
    default_path = Path(os.getcwd()).parent.parent.parent / "config.yaml"
    if default_path.exists():
        return str(default_path)

    # Second option
    default_path = Path(os.getcwd()) / "config.yaml"
    if default_path.exists():
        return str(default_path)

    raise FileNotFoundError(
        "Configuration file not found. Provide it via the mnemorai_CONFIG_PATH "
        "environment variable, or use the --config argument"
    )


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    Returns
    -------
    dict
        The parsed configuration data as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.full_load(f)


config_path = load_config_path()
config = load_config(config_path)
