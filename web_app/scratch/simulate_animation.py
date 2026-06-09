import os
from PIL import Image

def simulate():
    # Base container is 247x282
    base = Image.new('RGBA', (247, 282), (255, 255, 255, 0))
    
    # Load empty basket image and paste it
    empty_img = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    base.alpha_composite(empty_img, (0, 0))
    
    # CSS positions of gadgets:
    # watch: top 102px, left 58px, width 32px
    # camera: top 102px, left 90px, width 50px
    # tablet: top 102px, left 140px, width 38px
    # laptop: top 60px, left 80px, width 62px
    # phone: top 60px, left 136px, width 24px
    # headphones: top 60px, left 160px, width 42px
    
    gadgets = {
        "watch": ("imagens/gadgets/watch.png", 58, 102, 32),
        "camera": ("imagens/gadgets/camera.png", 90, 102, 50),
        "tablet": ("imagens/gadgets/tablet.png", 140, 102, 38),
        "laptop": ("imagens/gadgets/laptop.png", 80, 60, 62),
        "phone": ("imagens/gadgets/phone.png", 136, 60, 24),
        "headphones": ("imagens/gadgets/headphones.png", 160, 60, 42)
    }
    
    for name, (path, left, top, width) in gadgets.items():
        g_img = Image.open(path).convert('RGBA')
        # resize to match CSS width (preserving aspect ratio)
        orig_w, orig_h = g_img.size
        height = int(orig_h * width / orig_w)
        g_img_resized = g_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Create a temp image of same size as base to composite
        temp = Image.new('RGBA', (247, 282), (255, 255, 255, 0))
        temp.paste(g_img_resized, (left, top))
        base.alpha_composite(temp)
        
    os.makedirs('scratch', exist_ok=True)
    base.save('scratch/simulated_result.png')
    print("Saved scratch/simulated_result.png")

if __name__ == "__main__":
    simulate()
