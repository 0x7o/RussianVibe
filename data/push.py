from datasets import Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm
import json
import os

data = {"image": [], "caption": []}
input_dir = "images/"

for image_path in tqdm(glob(os.path.join(input_dir, "*.jpg"))):
    image = Image.open(image_path)
    try:
        with open(image_path.replace(".jpg", ".json"), "r") as f:
            data["caption"].append(json.loads(f.read())["caption"])
    except Exception as e:
        print(e, image_path)
    data["image"].append(image)

dataset = Dataset.from_dict(data)
dataset.shuffle()
dataset.push_to_hub("0x7o/RussianVibe-data")
