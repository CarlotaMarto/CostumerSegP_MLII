import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])

def create_gradient_map(cmap_name, new_colors_hex, num_steps=256):
    old_cmap = plt.get_cmap(cmap_name)
    old_ramp = old_cmap(np.linspace(0, 1, num_steps))[:, :3] * 255.0
    
    # Create new ramp from hex colors
    new_ramp = np.zeros((num_steps, 3))
    n_colors = len(new_colors_hex)
    rgbs = [hex_to_rgb(h) for h in new_colors_hex]
    
    for i in range(num_steps):
        t = i / (num_steps - 1)
        # Find which segment of the new ramp we are in
        seg = t * (n_colors - 1)
        idx = int(seg)
        if idx >= n_colors - 1:
            new_ramp[i] = rgbs[-1]
        else:
            frac = seg - idx
            new_ramp[i] = rgbs[idx] * (1 - frac) + rgbs[idx+1] * frac
            
    return old_ramp, new_ramp

def recolor_heatmap(img_path, cmap_name, new_colors_hex, crop_top=60):
    if not os.path.exists(img_path):
        return
        
    img = Image.open(img_path).convert("RGB")
    if crop_top > 0:
        img = img.crop((0, crop_top, img.width, img.height))
        
    pixels = np.array(img, dtype=np.float32)
    old_ramp, new_ramp = create_gradient_map(cmap_name, new_colors_hex)
    
    orig_shape = pixels.shape
    pixels_flat = pixels.reshape(-1, 3)
    
    chunk_size = 10000
    new_pixels_flat = np.copy(pixels_flat)
    
    for i in range(0, len(pixels_flat), chunk_size):
        chunk = pixels_flat[i:i+chunk_size]
        diffs = chunk[:, np.newaxis, :] - old_ramp[np.newaxis, :, :]
        dists = np.sum(diffs**2, axis=2)
        best_match_idx = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        
        mask = min_dists < 3000 # Increased threshold to catch compressed colours better
        new_pixels_flat[i:i+chunk_size][mask] = new_ramp[best_match_idx[mask]]
        
    new_pixels = new_pixels_flat.reshape(orig_shape)
    
    img_out = Image.fromarray(np.clip(new_pixels, 0, 255).astype(np.uint8))
    
    # Pad image (don't over-pad if not necessary)
    padding_left = 10
    padding_top = 10
    padding_right = 10
    padding_bottom = 20
    
    new_width = img_out.width + padding_left + padding_right
    new_height = img_out.height + padding_top + padding_bottom
    new_img = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    new_img.paste(img_out, (padding_left, padding_top))
    
    new_img.save(img_path)
    print(f"Processed heatmap {os.path.basename(img_path)}")

charts_dir = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts"

# zscore_heatmap.png uses RdBu_r
recolor_heatmap(
    os.path.join(charts_dir, "zscore_heatmap.png"), 
    "RdBu_r",
    ["#6B7D7D", "#fcfbf8", "#9D5C4A"] # Slate Blue to Beige to Rust
)

# silhouette_grid.png uses RdYlGn
recolor_heatmap(
    os.path.join(charts_dir, "silhouette_grid.png"), 
    "RdYlGn",
    ["#9D5C4A", "#fcfbf8", "#8B9D83"] # Rust to Beige to Sage Green
)

# NB4 heatmaps: spend_heatmap.png (sequential, Blues)
recolor_heatmap(
    os.path.join(charts_dir, "spend_heatmap.png"), 
    "Blues",
    ["#fcfbf8", "#6B7D7D"] # Beige to Slate Blue
)

# behavioural_heatmap.png (diverging, RdBu_r)
recolor_heatmap(
    os.path.join(charts_dir, "behavioural_heatmap.png"), 
    "RdBu_r",
    ["#6B7D7D", "#fcfbf8", "#9D5C4A"]
)
