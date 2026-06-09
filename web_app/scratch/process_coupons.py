import os
from PIL import Image

cupoes_dir = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\cupoes"
out_dir = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\cupoes_transp"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for f in os.listdir(cupoes_dir):
    if f.endswith(".png"):
        img = Image.open(os.path.join(cupoes_dir, f))
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            # check if pixel is white or very close to white
            if item[0] > 240 and item[1] > 240 and item[2] > 240:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save(os.path.join(out_dir, f), "PNG")
        print(f"Processed {f}")
