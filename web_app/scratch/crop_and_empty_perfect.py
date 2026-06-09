import os
from PIL import Image

def process_images_perfect():
    # Load original image
    img = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    width, height = img.size
    
    # Primary beige background color of original
    bg_color = (247, 239, 233)
    # Red stick figure color
    red_color = (186, 68, 53)
    
    # Bounding boxes for each gadget: [x_min, y_min, x_max, y_max]
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
        c_width, c_height = cropped.size
        
        # Make background transparent in the cropped image
        for y in range(c_height):
            for x in range(c_width):
                pixel = cropped.getpixel((x, y))
                
                # Check if it is a beige background pixel
                is_bg = all(abs(pixel[i] - bg_color[i]) < 15 for i in range(3))
                
                # Special case: for the watch, also remove any red stick figure arm pixels
                is_red_arm = False
                if name == "watch":
                    dist_to_red = ((pixel[0] - red_color[0])**2 + 
                                   (pixel[1] - red_color[1])**2 + 
                                   (pixel[2] - red_color[2])**2)**0.5
                    if dist_to_red < 30:
                        is_red_arm = True
                
                if is_bg or is_red_arm:
                    cropped.putpixel((x, y), (0, 0, 0, 0))
                    
        cropped.save(f'imagens/gadgets/{name}.png')
        print(f"Saved cleaned {name}.png with size {cropped.size}")
        
    # Generate empty basket image
    empty_img = img.copy()
    
    # Erase gadgets from the main image (Y: 58 to 138, X: 77 to 210)
    for y in range(58, 138):
        for x in range(77, 210):
            if x < width and y < height:
                pixel = empty_img.getpixel((x, y))
                
                # To protect the person's head/neck/arm on the left (X < 85):
                # We only erase if it is not red
                if x < 85:
                    dist_to_red = ((pixel[0] - red_color[0])**2 + 
                                   (pixel[1] - red_color[1])**2 + 
                                   (pixel[2] - red_color[2])**2)**0.5
                    if dist_to_red >= 30:
                        # Erase (replace with background color)
                        empty_img.putpixel((x, y), (247, 239, 233, 255))
                else:
                    # Erase completely (replace with background color)
                    empty_img.putpixel((x, y), (247, 239, 233, 255))
                    
    # Now convert all beige background pixels of empty_img to transparent
    for y in range(height):
        for x in range(width):
            pixel = empty_img.getpixel((x, y))
            is_bg = all(abs(pixel[i] - bg_color[i]) < 15 for i in range(3))
            if is_bg:
                empty_img.putpixel((x, y), (0, 0, 0, 0))
                
    empty_img.save('imagens/seg_basket_person_empty.png')
    print("Saved perfect seg_basket_person_empty.png with transparent background")

if __name__ == "__main__":
    process_images_perfect()
