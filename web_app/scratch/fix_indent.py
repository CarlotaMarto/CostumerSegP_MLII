import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.read().split('\n')

new_lines = []
for line in lines:
    if line.lstrip().startswith('if _p.exists():') and len(line) - len(line.lstrip()) == 8:
        # Check if the previous line was `_p = IMAGENS_DIR ...` which has 4 spaces.
        # So we should strip 4 spaces from this line.
        new_lines.append(line[4:])
    else:
        new_lines.append(line)

with open(app_path, "w", encoding="utf-8") as f:
    f.write('\n'.join(new_lines))
print("Fixed indentation!")
