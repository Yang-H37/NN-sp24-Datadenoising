import os
from PIL import Image

input_dir = "dataset/long_png"
output_dir = "dataset/long_png_resized"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path)
    width, height = img.size
    left = (width - 256) / 2
    top = (height - 256) / 2
    right = (width + 256) / 2
    bottom = (height + 256) / 2
    img_cropped = img.crop((left, top, right, bottom))
    output_path = os.path.join(output_dir, filename)
    img_cropped.save(output_path)

print("Cropping complete!")
