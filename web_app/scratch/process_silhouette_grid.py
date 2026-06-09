import os
from PIL import Image

path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts\silhouette_grid.png"
if os.path.exists(path):
    img = Image.open(path).convert("RGBA")
    
    # Just crop top to remove title, then pad
    img = img.crop((0, 50, img.width, img.height))
    
    padding_left = 30
    padding_top = 20
    padding_right = 10
    padding_bottom = 10
    
    new_width = img.width + padding_left + padding_right
    new_height = img.height + padding_top + padding_bottom
    new_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    new_img.paste(img, (padding_left, padding_top))
    new_img.save(path)
    print("Processed silhouette_grid.png")
