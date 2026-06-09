from PIL import Image

def check_watch_colors():
    watch = Image.open('imagens/gadgets/watch.png').convert('RGBA')
    w, h = watch.size
    
    red_figure_color = (186, 68, 53)
    
    matching_pixels = 0
    for y in range(h):
        for x in range(w):
            r, g, b, a = watch.getpixel((x, y))
            if a > 0:
                dist = ((r - red_figure_color[0])**2 + (g - red_figure_color[1])**2 + (b - red_figure_color[2])**2)**0.5
                if dist < 40:
                    matching_pixels += 1
                    
    print(f"Total visible pixels in watch: {sum(1 for y in range(h) for x in range(w) if watch.getpixel((x,y))[3] > 0)}")
    print(f"Pixels in watch close to red figure color (dist < 40): {matching_pixels}")

if __name__ == "__main__":
    check_watch_colors()
