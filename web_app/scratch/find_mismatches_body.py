from PIL import Image

def find_mismatches_body():
    orig = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    empty = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    
    red_color = (186, 68, 53)
    w, h = orig.size
    
    missing_by_x = {}
    for y in range(h):
        for x in range(w):
            p_orig = orig.getpixel((x, y))
            dist_orig = ((p_orig[0] - red_color[0])**2 + (p_orig[1] - red_color[1])**2 + (p_orig[2] - red_color[2])**2)**0.5
            
            if dist_orig < 30:
                # This was a red pixel in original
                # Check empty image
                p_empty = empty.getpixel((x, y))
                dist_empty = ((p_empty[0] - red_color[0])**2 + (p_empty[1] - red_color[1])**2 + (p_empty[2] - red_color[2])**2)**0.5
                
                # Check if it's no longer red/visible in empty
                if p_empty[3] == 0 or dist_empty >= 30:
                    missing_by_x[x] = missing_by_x.get(x, 0) + 1
                    
    print("Missing red pixels grouped by X coordinate:")
    for x in sorted(missing_by_x.keys()):
        print(f"X={x}: {missing_by_x[x]} missing red pixels")

if __name__ == "__main__":
    find_mismatches_body()
