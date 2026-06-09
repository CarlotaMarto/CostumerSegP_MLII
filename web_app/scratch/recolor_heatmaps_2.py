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
        # Don't actually crop, just keep the image intact so it's not cropped recursively
        pass
        
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
        
        mask = min_dists < 3000 # Threshold to catch compressed colours better
        new_pixels_flat[i:i+chunk_size][mask] = new_ramp[best_match_idx[mask]]
        
    new_pixels = new_pixels_flat.reshape(orig_shape)
    img_out = Image.fromarray(np.clip(new_pixels, 0, 255).astype(np.uint8))
    img_out.save(img_path)
    print(f"Processed heatmap {os.path.basename(img_path)}")

charts_dir = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts"

# spend_heatmap.png (sequential, YlOrRd)
recolor_heatmap(
    os.path.join(charts_dir, "spend_heatmap.png"), 
    "YlOrRd",
    ["#fcfbf8", "#9D5C4A"] # Beige to Rust
)
