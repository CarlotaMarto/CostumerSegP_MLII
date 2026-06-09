import json

with open("preprocessing/01_eda_preprocessing.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(7, 18):
    if idx >= len(nb.get("cells", [])):
        break
    cell = nb["cells"][idx]
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    print(f"=== Cell {idx} ({cell_type}) ===")
    print(source)
    print("\n" + "="*80 + "\n")
