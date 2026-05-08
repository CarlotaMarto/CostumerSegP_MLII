import pandas as pd

def get_missing_percent(df):
    """
    Calcula a percentagem de valores em falta para cada coluna do DataFrame.
    Retorna um DataFrame ordenado de forma decrescente.
    """
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    # Criar DataFrame apenas com a percentagem
    report = pd.DataFrame({
        'Column': missing_percent.index,
        'Missing_Percent': missing_percent.values
    })
    
    # Formatar e ordenar
    report['Missing_Percent'] = report['Missing_Percent'].round(2)
    report = report.sort_values('Missing_Percent', ascending=False)
    
    return report