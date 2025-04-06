import json

from mnemorai.constants.config import config

with open(config.get("G2P").get("LANGUAGE_JSON")) as f:
    G2P_LANGCODES = json.load(f)
G2P_LANGUAGES: dict = dict(map(reversed, G2P_LANGCODES.items()))
