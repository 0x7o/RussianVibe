# Russian Vibe
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://hf.co/spaces/0x7o/RussianVibe-1.0)

Fine-tuning Stable Diffusion XL to generate Russian corners.

## 🧨 Use with Diffusers
```bash
$ pip3 install diffusers
```
```python
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.load_lora_weights("0x7o/RussianVibe-XL-v2.0")
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "The sun is setting through a window, casting a warm glow on the cityscape beyond. The sun casts a warm orange glow on the buildings in the distance, creating a beautiful and serene atmosphere."
image = pipe(prompt, num_inference_steps=30, guidance_scale=5.0, negative_prompt="bad quality, painting, art").images[0]

image.save("output.png")
```

## Dataset
Download images from VK publics into one folder:
```bash
$ VK_TOKEN=... python3 data/dump_vk.py
```
Then generate photo captions using the [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) LLM (Note: This code requires Nvidia GPUs with at least 8 GB of video memory to run):
```bash
$ python3 data/generate_captions.py
```
Push the dataset to the Hugging Face hub:
```bash
$ python3 data/push.py
```
