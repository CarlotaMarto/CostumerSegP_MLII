import pandas as pd
from datetime import datetime


def get_missing_percent(df):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    report = pd.DataFrame({
        'Column': missing_percent.index,
        'Missing_Percent': missing_percent.values
    })
    
    report['Missing_Percent'] = report['Missing_Percent'].round(2)
    report = report.sort_values('Missing_Percent', ascending=False)
    
    return report

def calculate_percentage(df, condition):
    return (condition.sum() / len(df)) * 100


def get_invalid_years(df, year_col='year_first_transaction'):
    current_year = datetime.now().year
    invalid_rows = df[df[year_col] > current_year]
    return invalid_rows

def get_education_info(row):
    name = str(row['customer_name'])
    low_name = name.lower()
    
    if 'bsc.' in low_name:
        level = 1
        clean_name = name.replace('Bsc.', '').replace('bsc.', '').strip()
    elif 'msc.' in low_name:
        level = 2
        clean_name = name.replace('Msc.', '').replace('msc.', '').strip()
    elif 'phd.' in low_name:
        level = 3
        clean_name = name.replace('Phd.', '').replace('phd.', '').strip()
    else:
        level = 0
        clean_name = name.strip()
        
    return pd.Series([level, clean_name])