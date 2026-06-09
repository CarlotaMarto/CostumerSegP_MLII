import os
from PIL import Image

def analyze():
    img = Image.open('imagens/seg_basket_person.png')
    width, height = img.size
    print(f"Image dimensions: {width}x{height}")
    
    bg_color = (246, 238, 232)
    
    cols, rows = 40, 40
    dx = width / cols
    dy = height / rows
    
    grid = []
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            x = int(c * dx + dx/2)
            y = int(r * dy + dy/2)
            if x < width and y < height:
                pixel = img.getpixel((x, y))
                # Check if the pixel color is close to the background color
                is_bg = all(abs(pixel[i] - bg_color[i]) < 5 for i in range(3))
                if is_bg:
                    row_str += "."
                else:
                    row_str += "#"
            else:
                row_str += "."
        grid.append(row_str)
        
    for idx, row in enumerate(grid):
        print(f"{idx:02d}: {row}")

if __name__ == "__main__":
    analyze()
