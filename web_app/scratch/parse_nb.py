import json

with open("preprocessing/01_eda_preprocessing.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        if "Missing" in text or "Aggregation" in text or "Outliers" in text or "consensus" in text:
            print(f"--- Cell {i} (Markdown) ---")
            print(text[:800])
            print("\n")
    elif cell["cell_type"] == "code":
        text = "".join(cell["source"])
        if "utils." in text or "missing" in text or "outlier" in text:
            print(f"--- Cell {i} (Code) ---")
            print(text[:800])
            print("\n")
