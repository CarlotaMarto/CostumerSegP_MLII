from PIL import Image

def compare():
    img_orig = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    img_empty = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    
    w, h = img_orig.size
    print(f"Original size: {w}x{h}")
    print(f"Empty size: {img_empty.size}")
    
    # Check if there are any transparent/white areas in the stick figure area in both images
    # Stick figure X bounds in original were around 46 to 170.
    # Let's count non-transparent, non-beige pixels in the head region (X: 46-77, Y: 100-140)
    print("--- Original ---")
    for y in range(100, 140):
        non_bg_orig = []
        for x in range(46, 85):
            p = img_orig.getpixel((x, y))
            # not beige
            if not (abs(p[0]-246)<15 and abs(p[1]-238)<15 and abs(p[2]-232)<15):
                non_bg_orig.append(x)
        if non_bg_orig:
            print(f"Y={y}: non-beige X coords: {min(non_bg_orig)} to {max(non_bg_orig)}")

    print("--- Empty ---")
    for y in range(100, 140):
        non_bg_empty = []
        for x in range(46, 85):
            p = img_empty.getpixel((x, y))
            # not transparent
            if p[3] > 0:
                non_bg_empty.append(x)
        if non_bg_empty:
            print(f"Y={y}: non-transparent X coords: {min(non_bg_empty)} to {max(non_bg_empty)}")

if __name__ == "__main__":
    compare()
