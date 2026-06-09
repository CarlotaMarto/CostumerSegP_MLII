import pandas as pd
from pathlib import Path

BASE_DIR = Path("c:/Users/carlo/Documents/Semestre 4/ML 2/costumer_seg_app_cm")
df = pd.read_csv(BASE_DIR / "datasets" / "customer_info.csv")
print("Columns in customer_info.csv:")
print(df.columns.tolist())
if "education_level" in df.columns:
    print("education_level values:")
    print(df["education_level"].value_counts(dropna=False))
