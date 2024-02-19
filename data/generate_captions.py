from transformers import AutoProcessor, AutoModelForVision2Seq
from multiprocessing import Process
from glob import glob
from tqdm import tqdm
from PIL import Image
import json
import os
import torch

prompt = "Describe this image in detail:"
input_dir = "images/"


def worker(device_id: int, image_paths: list):
    torch.cuda.set_device(device_id)
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").cuda()
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    for image_path in tqdm(image_paths, desc=f"Device {device_id}"):
        json_file = image_path.replace(".jpg", ".json")

        if os.path.exists(json_file):
            continue

        with open(json_file, "w") as f:
            json.dump({"caption": get_caption(image_path, model, processor)}, f)


def get_caption(image_path: str, model, processor) -> str:
    image = Image.open(image_path)
    inputs = processor(text=f"<grounding>{prompt}", images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, entities = processor.post_process_generation(generated_text)
    return processed_text.replace(prompt, "").strip()


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    image_paths = glob(os.path.join(input_dir, "*.jpg"))
    split_image_paths = [image_paths[i::num_gpus] for i in range(num_gpus)]

    processes = []

    for i in range(num_gpus):
        p = Process(target=worker, args=(i, split_image_paths[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
