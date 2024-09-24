from huggingface_hub import hf_hub_download
from stable_diffusion_cpp import StableDiffusion

# Start by downloading the models
hf_hub_download(
    repo_id="aifoundry-org/FLUX.1-schnell-Quantized",
    filename="flux1-schnell-Q2_K.gguf",
    cache_dir="models",
)

# Vae
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    filename="ae.safetensors",
    cache_dir="models",
)

# Clip L
hf_hub_download(
    repo_id="comfyanonymous/flux_text_encoders",
    filename="clip_l.safetensors",
    cache_dir="models",
)

# T5XXL
hf_hub_download(
    repo_id="comfyanonymous/flux_text_encoders",
    filename="t5xxl_fp16.safetensors",
    cache_dir="models",
)

stable_diffusion = StableDiffusion(
    diffusion_model_path="models/models--aifoundry-org--FLUX.1-schnell-Quantized/snapshots/01c9555c34d40492cb4cbeb68af90a5c9e3ac4b4/flux1-schnell-Q2_K.gguf",  # In place of model_path
    clip_l_path="models/models--comfyanonymous--flux_text_encoders/snapshots/2f74b39c0606dae3b2196d79c18c2a40b71f3250/clip_l.safetensors",
    t5xxl_path="models/models--comfyanonymous--flux_text_encoders/snapshots/2f74b39c0606dae3b2196d79c18c2a40b71f3250/t5xxl_fp16.safetensors",
    vae_path="models/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/ae.safetensors",
)
output = stable_diffusion.txt_to_img(
    prompt="a lovely cat holding a sign says 'flux.cpp'",
    sample_steps=4,
    cfg_scale=1.0,  # a cfg_scale of 1 is recommended for FLUX
    sample_method="euler",  # euler is recommended for FLUX
)
print(type(output))
output.save("text2img_tests/flux_gguf.jpg")
