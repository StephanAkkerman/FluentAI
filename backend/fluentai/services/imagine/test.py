import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)
from diffusers import (
    SanaPipeline,
    SanaTransformer2DModel,
)
from transformers import AutoModel
from transformers import BitsAndBytesConfig as BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = AutoModel.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    cache_dir="models",
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = SanaTransformer2DModel.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    cache_dir="models",
)

pipeline = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
image = pipeline(prompt).images[0]
image.save("sana.png")
