from PIL import Image

def verify_watch_clean():
    watch = Image.open('imagens/gadgets/watch.png').convert('RGBA')
    w, h = watch.size
    
    red_color = (186, 68, 53)
    matching_pixels = 0
    for y in range(h):
        for x in range(w):
            r, g, b, a = watch.getpixel((x, y))
            if a > 0:
                dist = ((r - red_color[0])**2 + (g - red_color[1])**2 + (b - red_color[2])**2)**0.5
                if dist < 30:
                    matching_pixels += 1
                    
    print(f"Red pixels remaining in watch.png: {matching_pixels}")

if __name__ == "__main__":
    verify_watch_clean()
