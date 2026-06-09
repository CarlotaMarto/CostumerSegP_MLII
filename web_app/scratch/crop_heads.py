from PIL import Image

def crop_heads():
    orig = Image.open('imagens/seg_basket_person.png').convert('RGBA')
    empty = Image.open('imagens/seg_basket_person_empty.png').convert('RGBA')
    
    # Bounding box around the head
    box = (45, 100, 80, 130)
    
    orig_head = orig.crop(box)
    empty_head = empty.crop(box)
    
    orig_head.save('scratch/head_orig.png')
    empty_head.save('scratch/head_empty.png')
    print("Saved head crops to scratch/head_orig.png and scratch/head_empty.png")

if __name__ == "__main__":
    crop_heads()
