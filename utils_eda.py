import pandas as pd

def get_missing_percent(df):
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    report = pd.DataFrame({
        'Column': missing_percent.index,
        'Missing_Percent': missing_percent.values
    })
    
    report['Missing_Percent'] = report['Missing_Percent'].round(2)
    report = report.sort_values('Missing_Percent', ascending=False)
    
    return report