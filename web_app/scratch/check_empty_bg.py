from PIL import Image

def check_bg():
    img = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    width, height = img.size
    
    bg_color = (246, 238, 232)
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
                
    print(f"Total pixels: {width * height}")
    print(f"Transparent pixels: {transparent_count}")
    print(f"Beige pixels remaining (diff < 15): {beige_count}")
    print(f"Other pixels: {other_count}")

if __name__ == "__main__":
    check_bg()
