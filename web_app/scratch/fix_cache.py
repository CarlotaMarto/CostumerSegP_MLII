import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add the load_data function after `import pandas as pd`
if "def load_csv_data(filename):" not in content:
    load_func = """
@st.cache_data
def load_csv_data(filename):
    return pd.read_csv(BASE_DIR / "datasets" / filename)
"""
    # Find the first occurrence of `import pandas as pd` or similar to insert below it
    insert_pos = content.find("BASE_DIR = Path(__file__).resolve().parent")
    if insert_pos != -1:
        end_of_line = content.find("\n", insert_pos) + 1
        content = content[:end_of_line] + load_func + content[end_of_line:]

# Replace pd.read_csv(BASE_DIR / "datasets" / "...") with load_csv_data("...")
# We can use regex for this
# pattern: (?:pd|_pd)\.read_csv\(\s*BASE_DIR\s*\/\s*"datasets"\s*\/\s*"([^"]+)"\s*\)
pattern = r"(?:pd|_pd)\.read_csv\(\s*BASE_DIR\s*/\s*\"datasets\"\s*/\s*\"([^\"]+)\"\s*\)"
replacement = r'load_csv_data("\1")'

new_content = re.sub(pattern, replacement, content)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(new_content)
print("Replaced all pd.read_csv with cached load_csv_data.")
