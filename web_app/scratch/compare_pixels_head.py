from PIL import Image

def compare_head_pixels():
    orig = Image.open('scratch/head_orig.png').convert('RGBA')
    empty = Image.open('scratch/head_empty.png').convert('RGBA')
    
    w, h = orig.size
    bg_color = (247, 239, 233)
    
    mismatches = 0
    for y in range(h):
        for x in range(w):
            p_orig = orig.getpixel((x, y))
            p_empty = empty.getpixel((x, y))
            
            is_bg_orig = all(abs(p_orig[i] - bg_color[i]) < 15 for i in range(3))
            
            if not is_bg_orig:
                # If it's not background in original, it should be visible (alpha > 0) in empty
                if p_empty[3] == 0:
                    print(f"Pixel at ({x}, {y}) was non-background in original {p_orig} but is transparent in empty!")
                    mismatches += 1
                elif p_orig[:3] != p_empty[:3]:
                    print(f"Pixel at ({x}, {y}) color changed from {p_orig} to {p_empty}!")
                    mismatches += 1
                    
    print(f"Total mismatches: {mismatches}")

if __name__ == "__main__":
    compare_head_pixels()
