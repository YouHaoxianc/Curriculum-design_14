import os
from PIL import Image

for image in os.listdir('images'):
    img = Image.open(f'images/{image}')
    w = round(384 * img.size[0] / img.size[1])
    img = img.resize((w, 384), Image.Resampling.BICUBIC)
    img.save(f'images/{image}')
