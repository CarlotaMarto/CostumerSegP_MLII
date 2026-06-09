from PIL import Image
import os

paths = [
    r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts\scaler_comparison.png",
    r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts\elbow_silhouette.png"
]

for path in paths:
    if os.path.exists(path):
        img = Image.open(path).convert("RGBA")
        
        # Add 25 pixels padding on left and top, 15 on right and bottom
        padding_left = 30
        padding_top = 20
        padding_right = 10
        padding_bottom = 10
        
        new_width = img.width + padding_left + padding_right
        new_height = img.height + padding_top + padding_bottom
        
        # Create a white background image (since the graphs typically have white bg)
        # Using transparent (255,255,255,0) might make the padding look weird if it has a border.
        # But Streamlit renders it on a white card anyway.
        new_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
        new_img.paste(img, (padding_left, padding_top))
        new_img.save(path)
        print(f"Padded {os.path.basename(path)}")
