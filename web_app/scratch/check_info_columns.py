import pandas as pd
from pathlib import Path

def check_cols():
    df = pd.read_csv('datasets/customer_info.csv')
    print("Columns in customer_info.csv:")
    print(list(df.columns))
    print("\nFirst row:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    check_cols()
