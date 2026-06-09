import os
import numpy as np
from PIL import Image

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

old_colors = np.array([
    [31, 119, 180],   # 0: blue
    [255, 127, 14],   # 1: orange
    [44, 160, 44],    # 2: green
    [214, 39, 40],    # 3: red
    [148, 103, 189],  # 4: purple
    [140, 86, 75],    # 5: brown
    [227, 119, 194],  # 6: pink
    [127, 127, 127],  # 7: gray
    [188, 189, 34],   # 8: olive
    [23, 190, 207],   # 9: cyan
])

segment_colors_hex = {
    0: "#8B9D83", # Sage Green
    1: "#D3A975", # Warm Sand
    2: "#B07D62", # Terracotta
    3: "#6B7D7D", # Slate Blue/Grey
    4: "#E0C3A0", # Light Beige
    5: "#9D5C4A", # Rust
    6: "#C5A880", # Camel
    7: "#5C6B5D"  # Forest Green
}
new_colors = np.array([hex_to_rgb(segment_colors_hex[i]) for i in range(8)])

def process_image(img_path, crop_top=60):
    if not os.path.exists(img_path):
        return
        
    img = Image.open(img_path).convert("RGB")
    
    if crop_top > 0:
        img = img.crop((0, crop_top, img.width, img.height))
        
    pixels = np.array(img, dtype=np.float32)
    max_c = np.max(pixels, axis=2)
    min_c = np.min(pixels, axis=2)
    saturation = max_c - min_c
    
    mask = saturation > 15
    colored_pixels = pixels[mask]
    
    if len(colored_pixels) > 0:
        cp_norm = colored_pixels / np.linalg.norm(colored_pixels, axis=1, keepdims=True)
        tab10_norm = old_colors / np.linalg.norm(old_colors, axis=1, keepdims=True)
        
        similarity = np.dot(cp_norm, tab10_norm.T)
        best_match = np.argmax(similarity, axis=1)
        
        new_pixels = np.zeros_like(colored_pixels)
        for i in range(8):
            cluster_mask = (best_match == i)
            if not np.any(cluster_mask): continue
            cp = colored_pixels[cluster_mask]
            old_c = old_colors[i]
            new_c = new_colors[i]
            alpha = np.mean((255.0 - cp) / (255.0 - old_c + 1e-5), axis=1, keepdims=True)
            alpha = np.clip(alpha, 0, 1)
            new_pixels[cluster_mask] = alpha * new_c + (1 - alpha) * 255.0
            
        for i in [8, 9]:
            cluster_mask = (best_match == i)
            if np.any(cluster_mask):
                cp = colored_pixels[cluster_mask]
                old_c = old_colors[i]
                new_c = new_colors[i - 8]
                alpha = np.mean((255.0 - cp) / (255.0 - old_c + 1e-5), axis=1, keepdims=True)
                alpha = np.clip(alpha, 0, 1)
                new_pixels[cluster_mask] = alpha * new_c + (1 - alpha) * 255.0
                
        pixels[mask] = new_pixels
        
    img = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    
    padding_left = 30
    padding_top = 20
    padding_right = 10
    padding_bottom = 10
    
    new_width = img.width + padding_left + padding_right
    new_height = img.height + padding_top + padding_bottom
    new_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    new_img.paste(img, (padding_left, padding_top))
    
    new_img.save(img_path)
    print(f"Processed {os.path.basename(img_path)}")

charts_dir = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts"
images_to_process = [
    "radar_individual.png", 
    "radar_combined.png", 
    "feature_barplots.png"
]

for img_name in images_to_process:
    process_image(os.path.join(charts_dir, img_name), crop_top=60)
