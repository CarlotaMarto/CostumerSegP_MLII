import os
import pytesseract
from PIL import Image

cupoes_dir = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\cupoes"
for f in os.listdir(cupoes_dir):
    if f.endswith(".png"):
        print(f"--- {f} ---")
        try:
            text = pytesseract.image_to_string(Image.open(os.path.join(cupoes_dir, f)))
            print(text.strip())
        except Exception as e:
            print("Error:", e)
