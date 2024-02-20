from datasets import Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm
import json
import os

data = {"image": [], "caption": []}
input_dir = "images/"
for image_path in tqdm(glob(os.path.join(input_dir, "*.jpg"))):
    try:
        with Image.open(image_path) as image:
            with open(image_path.replace(".jpg", ".json"), "r") as f:
                data["caption"].append(json.loads(f.read())["caption"])
            data["image"].append(image.copy())
    except Exception as e:
        print(e, image_path)
        continue

dataset = Dataset.from_dict(data)
dataset = dataset.shuffle()
dataset.push_to_hub("0x7o/RussianVibe-data")