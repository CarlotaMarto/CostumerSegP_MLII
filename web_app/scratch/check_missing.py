import pandas as pd
from pathlib import Path

BASE_DIR = Path("c:/Users/carlo/Documents/Semestre 4/ML 2/costumer_seg_app_cm")
df = pd.read_csv(BASE_DIR / "datasets" / "customer_info.csv")
print("Missing values in raw customer_info:")
print(df.isna().sum())
