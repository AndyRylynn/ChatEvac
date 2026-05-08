"""
Composite white background onto transparent PNG images.
Processes all .png files in the img/ directory.
"""
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "img")

for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith(".png"):
        continue

    fpath = os.path.join(IMG_DIR, fname)
    img = Image.open(fpath)

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        background.save(fpath, "PNG")
        print(f"  Composited white bg: {fname}")
    elif img.mode == "RGB":
        print(f"  Already RGB (no alpha): {fname}")
    else:
        # Convert palette/grayscale to RGB just in case
        img = img.convert("RGB")
        img.save(fpath, "PNG")
        print(f"  Converted to RGB: {fname}")

print("Done.")
