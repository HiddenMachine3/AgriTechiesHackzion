import os
from PIL import Image

directory1 = r"C:\Projects\AgriTechiesHackzion\data\ripeness_classification\apple\trainto_squares\ripe" # "../data/ripeness_classification/apple/trainto_squares/ripe"
directory2 = r"C:\Projects\AgriTechiesHackzion\data\ripeness_classification\apple\trainto_squares\unripe" # "../data/ripeness_classification/apple/trainto_squares/ripe"

def resize_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            resized_image = image.resize((256, 256))
            resized_image.save(image_path)
# resize_images(directory1)
# resize_images(directory2)