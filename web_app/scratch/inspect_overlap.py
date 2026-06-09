from PIL import Image

def inspect_overlap():
    empty = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    simulated = Image.open('scratch/simulated_result.png').convert('RGBA')
    
    # Head region
    box = (45, 100, 80, 130)
    
    # Count red pixels in both in the head region
    def count_red(img):
        count = 0
        for y in range(100, 130):
            for x in range(45, 80):
                r, g, b, a = img.getpixel((x, y))
                dist = ((r - 186)**2 + (g - 68)**2 + (b - 53)**2)**0.5
                if dist < 30 and a > 0:
                    count += 1
        return count
        
    print(f"Red pixels in empty head region: {count_red(empty)}")
    print(f"Red pixels in simulated head region: {count_red(simulated)}")

if __name__ == "__main__":
    inspect_overlap()
