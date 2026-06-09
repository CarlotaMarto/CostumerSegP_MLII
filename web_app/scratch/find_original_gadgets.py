import os
from PIL import Image

def find_elements():
    img = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    width, height = img.size
    
    # Background color is around (247, 239, 233)
    bg_color = (247, 239, 233)
    
    # Let's create a copy where we can mark things
    debug_img = img.copy()
    
    # We want to identify the exact bounding box of the red person, the brown basket, and each gadget.
    # The red person is reddish: R > 150, G < 100, B < 100
    # The brown basket/gadgets are brownish/greenish.
    # Let's write out a grid of non-background pixels to see what is where.
    for y in range(0, height, 5):
        row_str = f"Y={y:03d}: "
        for x in range(0, width, 5):
            pixel = img.getpixel((x, y))
            is_bg = all(abs(pixel[i] - bg_color[i]) < 10 for i in range(3))
            if is_bg:
                row_str += " "
            else:
                # check if reddish
                r, g, b, a = pixel
                if r > 150 and g < 100 and b < 100:
                    row_str += "P" # Person
                else:
                    row_str += "G" # Gadget/Basket
        print(row_str)

if __name__ == "__main__":
    find_elements()
