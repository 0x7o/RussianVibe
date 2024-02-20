from datasets import Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm
import json
import os

input_dir = "images/"


def gen():
    for image_path in tqdm(glob(os.path.join(input_dir, "*.jpg"))):
        with Image.open(image_path) as image:
            data = {}
            with open(image_path.replace(".jpg", ".json"), "r") as f:
                data["caption"] = json.loads(f.read())["caption"]
            data["image"] = image
            yield data


dataset = Dataset.from_generator(gen)
dataset = dataset.shuffle()
dataset.push_to_hub("0x7o/RussianVibe-data")
