from PIL import Image

def inspect_original_arm():
    img = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    
    print("Coordinates of pixels in Y=130 to 140, X=75 to 85:")
    for y in range(130, 140):
        row_str = f"Y={y}: "
        for x in range(75, 85):
            r, g, b, a = img.getpixel((x, y))
            # check if beige background
            bg_color = (247, 239, 233)
            is_bg = all(abs(img.getpixel((x,y))[i] - bg_color[i]) < 10 for i in range(3))
            
            if is_bg:
                row_str += "  [BG]  "
            else:
                row_str += f"({r:3d},{g:3d},{b:3d})"
        print(row_str)

if __name__ == "__main__":
    inspect_original_arm()
