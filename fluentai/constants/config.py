import os

import yaml

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
with open(config_path, encoding="utf-8") as f:
    config: dict = yaml.full_load(f)
