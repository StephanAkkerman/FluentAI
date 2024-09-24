import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
from transformers import CLIPTextModel, T5EncoderModel

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/argmaxinc/mlx-FLUX.1-schnell-4bit-quantized/blob/main/flux-schnell-4bit-quantized.safetensors",
    torch_dtype=dtype,
    cache_dir="models",
)
quantize(transformer, weights=qfloat8)
freeze(transformer)

text_encoder_2 = T5EncoderModel.from_pretrained(
    bfl_repo,
    subfolder="text_encoder_2",
    torch_dtype=dtype,
    cache_dir="models",
)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline.from_pretrained(
    bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype
)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("flux-fp8-dev.png")
