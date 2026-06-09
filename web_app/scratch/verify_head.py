from PIL import Image

def verify_head():
    img_orig = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    img_empty = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    
    # Let's count red pixels (dist < 30 from (186, 68, 53)) in the head region (X: 46-77, Y: 100-128)
    def count_red(img):
        count = 0
        for y in range(100, 128):
            for x in range(46, 77):
                r, g, b, a = img.getpixel((x, y))
                dist = ((r - 186)**2 + (g - 68)**2 + (b - 53)**2)**0.5
                if dist < 30 and a > 0:
                    count += 1
        return count
        
    print(f"Original head red pixels: {count_red(img_orig)}")
    print(f"Empty head red pixels: {count_red(img_empty)}")

if __name__ == "__main__":
    verify_head()
