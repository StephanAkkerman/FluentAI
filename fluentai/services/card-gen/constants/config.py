import os

import yaml

config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, encoding="utf-8") as f:
    config: dict = yaml.full_load(f)

weights = config["WEIGHTS"]
total_weight = sum(weights.values())
weights_percentages = {
    factor: (weight / total_weight) * 100 for factor, weight in weights.items()
}
