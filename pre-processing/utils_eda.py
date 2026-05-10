import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import KNNImputer

# --- MISSING VALUES & DATA INTEGRITY ---

def get_missing_percent(df):
    """Returns a dataframe with the percentage of missing values per column."""
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    report = pd.DataFrame({
        'Column': missing_percent.index,
        'Missing_Percent': missing_percent.values
    })
    
    report['Missing_Percent'] = report['Missing_Percent'].round(2)
    report = report.sort_values('Missing_Percent', ascending=False)
    
    return report

def calculate_percentage(df, condition):
    """Calculates the percentage of rows that meet a specific condition."""
    return (condition.sum() / len(df)) * 100

def get_invalid_years(df, year_col='year_first_transaction'):
    """Identifies rows where the transaction year is in the future."""
    current_year = datetime.now().year
    invalid_rows = df[df[year_col] > current_year]
    return invalid_rows

# --- DATA TRANSFORMATIONS ---

def get_education_info(row):
    """Extracts education level (0-3) and cleans the customer name."""
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

def get_missing_report(df):
    """Returns a dataframe with the count and percentage of missing values."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    report = pd.DataFrame({'Missing Count': missing_count, 'Percentage (%)': missing_pct})
    return report[report['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False)


def apply_cyclic_transformation(df, col, max_val=24):
    # Garante que a coluna existe e não tem NaNs (ou trata-os)
    if col in df.columns:
        # Criar cópia para evitar SettingWithCopyWarning dependendo de como o df foi filtrado
        temp_col = pd.to_numeric(df[col], errors='coerce')
        
        df[f'{col}_sin'] = np.sin(2 * np.pi * temp_col / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * temp_col / max_val)
    
    return df


# --- ADVANCED PREPROCESSING ---

def apply_knn_imputation(df, n_neighbors=5):
    """Applies KNN Imputation to numerical columns and returns the updated dataframe."""
    # Select only numeric columns for KNN
    numeric_df = df.select_dtypes(include=[np.number])
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Perform imputation (returns a numpy array)
    imputed_data = imputer.fit_transform(numeric_df)
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)
    
    # Update the original dataframe (keeping non-numeric columns intact)
    df_new = df.copy()
    df_new.update(imputed_df)
    return df_new

def get_high_correlations(df, threshold=0.7):
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr = [(column, row, upper[column][row]) 
                 for row in upper.index 
                 for column in upper.columns 
                 if upper[column][row] > threshold]
    
    result = pd.DataFrame(high_corr, columns=['Variable 1', 'Variable 2', 'Correlation'])
    return result.sort_values(by='Correlation', ascending=False)
# --- VISUALIZATION ---

def plot_missing_heatmap(df, title="Missing Values Heatmap"):
    """Plots a heatmap to visualize the location of missing data."""
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(title)
    plt.show()

def cor_heatmap(corr_matrix, color="#1B4F72"):
    # Aumentamos o tamanho significativamente (20x15)
    plt.figure(figsize=(20, 15))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.light_palette(color, as_cmap=True)
    
    # annot_kws ajusta o tamanho da letra dentro dos quadrados
    sns.heatmap(corr_matrix, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap, 
                center=0, 
                square=True, 
                linewidths=.5, 
                annot_kws={"size": 8},
                cbar_kws={"shrink": .5})
    
    plt.title('Correlation Heatmap', fontsize=20, fontweight='bold', pad=25)
    plt.xticks(rotation=45, ha='right') # Roda os nomes das colunas para não chocarem
    plt.show()

def handle_extreme_outliers(df, columns, strategy='cap'):
    """
    Handles only extreme outliers using the 3.0 * IQR rule.
    Extreme outliers are values beyond [Q1 - 3*IQR, Q3 + 3*IQR].
    """
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # O fator 3.0 define o limite para "Extremos"
        lower_bound = Q1 - (3.0 * IQR)
        upper_bound = Q3 + (3.0 * IQR)
        
        if strategy == 'cap':
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, 
                                   np.where(df_out[col] < lower_bound, lower_bound, df_out[col]))
    return df_out