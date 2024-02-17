from transformers import AutoProcessor, AutoModelForVision2Seq
from glob import glob
from PIL import Image
import json
import os

model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").cuda()
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

prompt = "Describe this image in detail:"
input_dir = "images/"


def get_caption(image_path: str) -> str:
    image = Image.open(image_path)
    inputs = processor(text=f"<grounding>{prompt}", images=image, return_tensors="pt")
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, entities = processor.post_process_generation(generated_text)
    return processed_text.replace(prompt, "").strip()


if __name__ == "__main__":
    for image_path in glob(os.path.join(input_dir, "*.jpg")):
        caption = get_caption(image_path)

        with open(image_path.replace(".jpg", ".json"), "w") as f:
            json.dump({"caption": caption}, f)
