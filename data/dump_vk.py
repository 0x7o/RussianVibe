import requests
import os
import time

from typing import List
from tqdm import tqdm

vk_token = os.environ["VK_TOKEN"]
vk_clubs = [-201677255, -101842889, -155920876, -97980111, -201677255]
output_dir = "images/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def download_image(image_url: str) -> None:
    r = requests.get(image_url)
    file_name = os.path.join(output_dir, image_url.split("?")[0].split("/")[-1])

    if os.path.exists(file_name):
        return

    with open(
        os.path.join(output_dir, image_url.split("?")[0].split("/")[-1]), "wb"
    ) as f:
        f.write(r.content)


def get_images(club_id: int, n=100) -> List[str]:
    images = []
    processed = 0
    sleep = 0

    while len(images) < n or n == -1:
        r = requests.get(
            "https://api.vk.com/method/wall.get",
            params={
                "access_token": vk_token,
                "owner_id": club_id,
                "offset": processed,
                "count": 100,
                "v": 5.199,
            },
        ).json()
        response = r.get("response")

        if not response:
            sleep += 5
            print(f"[Error!] Sleeping {sleep} s. {r}")
            time.sleep(sleep)
            continue

        if not response["items"]:
            return images

        for item in response["items"]:
            attachments = item["attachments"]

            if not attachments:
                continue

            for attach in attachments:
                if attach["type"] != "photo":
                    continue

                for size in attach["photo"]["sizes"]:
                    if size["type"] != "w":
                        continue

                    images.append(size["url"])
                    break

        print(f"Getting {len(images)} images from {processed} / {n} posts...")
        processed += 100

    return images


if __name__ == "__main__":
    for club in vk_clubs:
        for image in tqdm(get_images(club, -1), desc="Downloading images"):
            download_image(image)
