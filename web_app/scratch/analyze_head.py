from PIL import Image

def find_head():
    img = Image.open('imagens/seg_basket_person.png')
    img = img.convert('RGBA')
    width, height = img.size
    
    # Let's find all red pixels of the stick figure (e.g. R > 150, G < 100, B < 100)
    red_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            # The red color is around (186, 68, 53)
            # Let's use a distance threshold
            dist = ((r - 186)**2 + (g - 68)**2 + (b - 53)**2)**0.5
            if dist < 30:
                red_pixels.append((x, y))
                
    print(f"Total red pixels: {len(red_pixels)}")
    if red_pixels:
        min_x = min(p[0] for p in red_pixels)
        max_x = max(p[0] for p in red_pixels)
        min_y = min(p[1] for p in red_pixels)
        max_y = max(p[1] for p in red_pixels)
        print(f"Red pixels bounds: X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
        
        # Let's print out the red pixels for the upper part of the image (Y from min_y to min_y + 50)
        # to see the shape of the head and where it is located.
        for y in range(min_y, min_y + 40):
            row_str = ""
            for x in range(min_x, min_x + 60):
                r, g, b, a = img.getpixel((x, y))
                dist = ((r - 186)**2 + (g - 68)**2 + (b - 53)**2)**0.5
                if dist < 30:
                    row_str += "#"
                else:
                    # check if it's beige
                    bg_color = (246, 238, 232)
                    is_bg = all(abs(img.getpixel((x,y))[i] - bg_color[i]) < 15 for i in range(3))
                    if is_bg:
                        row_str += "."
                    else:
                        row_str += "o" # some other color (gadget, etc.)
            print(f"Y={y:03d}: {row_str}")

if __name__ == "__main__":
    find_head()
