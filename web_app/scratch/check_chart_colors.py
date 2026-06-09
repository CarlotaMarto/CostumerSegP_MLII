from PIL import Image
from collections import Counter
import os

img_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts\pca_projection.png"
if not os.path.exists(img_path):
    print("Image not found:", img_path)
    exit()

img = Image.open(img_path).convert("RGB")
pixels = list(img.getdata())

# Filter out white/black/grays
colored_pixels = []
for p in pixels:
    if max(p) - min(p) > 30:
        colored_pixels.append(p)

counts = Counter(colored_pixels)

print("Top 20 colors in PCA Projection:")
for color, count in counts.most_common(20):
    h = "#{:02x}{:02x}{:02x}".format(*color)
    print(f"Color {h} - Count {count}")
