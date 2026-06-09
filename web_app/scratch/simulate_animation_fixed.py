import os
from PIL import Image

def simulate_fixed():
    base = Image.new('RGBA', (247, 282), (255, 255, 255, 0))
    empty_img = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    base.alpha_composite(empty_img, (0, 0))
    
    # Original exact coordinates of gadgets:
    # watch: left 77px, top 102px, width 25px
    # camera: left 102px, top 102px, width 40px
    # tablet: left 142px, top 102px, width 38px
    # laptop: left 80px, top 58px, width 62px
    # phone: left 136px, top 58px, width 24px
    # headphones: left 160px, top 58px, width 45px
    
    gadgets = {
        "watch": ("imagens/gadgets/watch.png", 77, 102, 25),
        "camera": ("imagens/gadgets/camera.png", 102, 102, 40),
        "tablet": ("imagens/gadgets/tablet.png", 142, 102, 38),
        "laptop": ("imagens/gadgets/laptop.png", 80, 58, 62),
        "phone": ("imagens/gadgets/phone.png", 136, 58, 24),
        "headphones": ("imagens/gadgets/headphones.png", 160, 58, 45)
    }
    
    for name, (path, left, top, width) in gadgets.items():
        g_img = Image.open(path).convert('RGBA')
        orig_w, orig_h = g_img.size
        height = int(orig_h * width / orig_w)
        g_img_resized = g_img.resize((width, height), Image.Resampling.LANCZOS)
        
        temp = Image.new('RGBA', (247, 282), (255, 255, 255, 0))
        temp.paste(g_img_resized, (left, top))
        base.alpha_composite(temp)
        
    base.save('scratch/simulated_result_fixed.png')
    
    # Let's check overlap
    def count_red(img):
        count = 0
        for y in range(100, 130):
            for x in range(45, 80):
                r, g, b, a = img.getpixel((x, y))
                dist = ((r - 186)**2 + (g - 68)**2 + (b - 53)**2)**0.5
                if dist < 30 and a > 0:
                    count += 1
        return count
        
    print(f"Red pixels in empty head region: {count_red(empty_img)}")
    print(f"Red pixels in fixed simulated head region: {count_red(base)}")

if __name__ == "__main__":
    simulate_fixed()
