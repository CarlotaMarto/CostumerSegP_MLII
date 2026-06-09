import os
from PIL import Image

def check_gadgets():
    bg_color = (246, 238, 232)
    gadget_dir = 'imagens/gadgets'
    if not os.path.exists(gadget_dir):
        print("Gadgets directory does not exist")
        return
        
    for filename in os.listdir(gadget_dir):
        if filename.endswith('.png'):
            path = os.path.join(gadget_dir, filename)
            img = Image.open(path).convert('RGBA')
            width, height = img.size
            beige_count = 0
            transparent_count = 0
            other_count = 0
            for y in range(height):
                for x in range(width):
                    r, g, b, a = img.getpixel((x, y))
                    if a == 0:
                        transparent_count += 1
                    elif all(abs(img.getpixel((x,y))[i] - bg_color[i]) < 15 for i in range(3)):
                        beige_count += 1
                    else:
                        other_count += 1
            print(f"{filename}: size={width}x{height}, transparent={transparent_count}, beige={beige_count}, other={other_count}")

if __name__ == "__main__":
    check_gadgets()
