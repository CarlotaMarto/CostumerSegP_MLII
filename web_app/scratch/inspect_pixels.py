import os
from PIL import Image

def inspect():
    img = Image.open('imagens/seg_basket_person.png')
    img = img.convert('RGBA')
    width, height = img.size
    print(f"Dimensions: {width}x{height}")
    
    # Let's count pixel colors to understand the palette
    colors = {}
    for y in range(height):
        for x in range(width):
            pixel = img.getpixel((x, y))
            colors[pixel] = colors.get(pixel, 0) + 1
            
    # Print the top 10 most common colors
    sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 colors:")
    for color, count in sorted_colors[:10]:
        print(f"Color: {color}, Count: {count}")

    # Let's find the bounding box of the red/orange stick figure
    # The figure is reddish. Let's look for pixels with high R compared to G and B,
    # or list some positions of reddish pixels.
    red_pixels = []
    for y in range(height):
        for x in range(width):
            r, g, b, a = img.getpixel((x, y))
            # Let's see if it's reddish (e.g. R > 150 and G < 100 and B < 100)
            if r > 150 and g < 100 and b < 100:
                red_pixels.append((x, y))
                
    if red_pixels:
        xs = [p[0] for p in red_pixels]
        ys = [p[1] for p in red_pixels]
        print(f"Red figure bounding box: X: {min(xs)} to {max(xs)}, Y: {min(ys)} to {max(ys)}")
    else:
        print("No red pixels found with threshold R > 150, G < 100, B < 100")

if __name__ == "__main__":
    inspect()
