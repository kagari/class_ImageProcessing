from PIL import Image, ImageOps
import numpy as np

if __name__ == "__main__":
    img = Image.open("../data/city-3021474_1920.jpg")

    img_mirror = ImageOps.mirror(img)

    with open("../data/task3/city_mirror.jpg", "w") as f:
        img_mirror.save(f, format="JPEG")
