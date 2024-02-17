from datasets import Dataset
from glob import glob
from PIL import Image
import json
import os

data = {"image": [], "caption": []}
input_dir = "images/"

for image_path in glob(os.path.join(input_dir, "*.jpg")):
    image = Image.open(image_path)
    data["image"].append(image)
    data["caption"].append(json.loads(image_path.replace(".jpg", ".json"))["caption"])

dataset = Dataset.from_dict(data)
dataset.shuffle()
dataset.push_to_hub("0x7o/RussianVibe-data")
