from PIL import Image

def verify_entire_body():
    orig = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    empty = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    
    red_color = (186, 68, 53)
    
    def count_red(img):
        count = 0
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = img.getpixel((x, y))
                # If transparent, don't count
                if a == 0:
                    continue
                dist = ((r - red_color[0])**2 + (g - red_color[1])**2 + (b - red_color[2])**2)**0.5
                if dist < 30:
                    count += 1
        return count
        
    orig_count = count_red(orig)
    empty_count = count_red(empty)
    
    print(f"Total red pixels in original image: {orig_count}")
    print(f"Total red pixels in empty basket image: {empty_count}")
    print(f"Difference: {orig_count - empty_count}")

if __name__ == "__main__":
    verify_entire_body()
