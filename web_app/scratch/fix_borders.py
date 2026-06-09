from PIL import Image
from pathlib import Path

img_path = Path(r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\seg_basket_person_empty.png")
if not img_path.exists():
    print("Image not found!")
    exit(1)

img = Image.open(img_path).convert("RGBA")
width, height = img.size
pixels = img.load()

count = 0
for y in range(height):
    for x in range(width):
        r, g, b, a = pixels[x, y]
        # If the pixel is pure white (or very close to pure white)
        if r == 255 and g == 255 and b == 255:
            pixels[x, y] = (0, 0, 0, 0)
            count += 1

img.save(img_path)
print(f"Replaced {count} white pixels with transparent pixels and saved the image.")
