from PIL import Image
from pathlib import Path

img_path = Path(r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\seg_basket_person_empty.png")
if not img_path.exists():
    print("Image not found!")
    exit(1)

img = Image.open(img_path)
print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")

# Check the borders
width, height = img.size
pixels = img.load()

# Let's inspect the right-most border and bottom-most border
print("Right border (last 5 columns, middle 10 rows):")
for y in range(height // 2 - 5, height // 2 + 5):
    row_pixels = [pixels[x, y] for x in range(width - 5, width)]
    print(f"y={y}: {row_pixels}")

print("\nBottom border (last 5 rows, middle 10 columns):")
for y in range(height - 5, height):
    col_pixels = [pixels[x, y] for x in range(width // 2 - 5, width // 2 + 5)]
    print(f"y={y}: {col_pixels}")
