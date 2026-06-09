import os
from PIL import Image

def process_images():
    img = Image.open('imagens/seg_basket_person.png')
    img = img.convert('RGBA')
    width, height = img.size
    bg_color = (246, 238, 232)
    
    # Bounding boxes for each gadget: [x_min, y_min, x_max, y_max]
    # watch starts at X = 77 to avoid cropping the person's head
    gadgets_boxes = {
        "laptop": [80, 58, 142, 102],
        "phone": [136, 58, 160, 102],
        "headphones": [160, 58, 205, 102],
        "watch": [77, 102, 102, 137],
        "camera": [102, 102, 142, 137],
        "tablet": [142, 102, 180, 137]
    }
    
    os.makedirs('imagens/gadgets', exist_ok=True)
    
    for name, box in gadgets_boxes.items():
        cropped = img.crop(box)
        
        # Make background transparent in the cropped image
        c_width, c_height = cropped.size
        for y in range(c_height):
            for x in range(c_width):
                pixel = cropped.getpixel((x, y))
                is_bg = all(abs(pixel[i] - bg_color[i]) < 15 for i in range(3))
                if is_bg:
                    cropped.putpixel((x, y), (0, 0, 0, 0))
                    
        cropped.save(f'imagens/gadgets/{name}.png')
        print(f"Saved {name}.png with size {cropped.size}")
        
    # Erase gadgets from the main image to make an empty basket
    empty_img = img.copy()
    
    # We will fill the rectangle containing the gadgets with the background color
    # X starts at 77 to protect the person's head/neck on the left
    for y in range(58, 138):
        for x in range(77, 210):
            if x < width and y < height:
                empty_img.putpixel((x, y), (246, 238, 232, 255))
                
    # Now, make the background of the base image transparent too
    for y in range(height):
        for x in range(width):
            pixel = empty_img.getpixel((x, y))
            is_bg = all(abs(pixel[i] - bg_color[i]) < 15 for i in range(3))
            if is_bg:
                empty_img.putpixel((x, y), (0, 0, 0, 0))
                
    empty_img.save('imagens/seg_basket_person_empty.png')
    print("Saved seg_basket_person_empty.png with transparent background")

if __name__ == "__main__":
    process_images()
