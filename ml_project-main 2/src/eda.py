import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import math
import numpy as np
import base64
from IPython.display import IFrame, display

def plot_missing_values(df, title='Count of Missing Values per Feature'):
    """
    Calculates and plots a horizontal bar chart of missing values in the dataset.
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
    plt.figure(figsize=(16, 5))
    sns.barplot(x=missing_data.values, y=missing_data.index, color='skyblue')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Missing Rows')
    plt.ylabel('')
    plt.show()



def plot_distribution(df):
    """
    Plots the distribution for all numeric columns in a clean grid layout.
    Automatically disables KDE for discrete variables (like counts).
    """
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'Int64']).columns
    
    n_cols = 7
    n_rows = math.ceil(len(numeric_cols) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2.5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        
        is_discrete = df[col].nunique() < 15
        
        sns.histplot(data=df, x=col, bins=30, color='skyblue', kde=not is_discrete, ax=axes[i])
        axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

def plot_pie_chart(df, variable, colors, legend, title_):

    """
    Creates a pie chart to visualize the distribution of a categorical variable.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - variable (str): The name of the column to plot.
    - colors (list): List of colors to use for the slices.
    - legend (list): List of labels for the legend.
    - title_ (str): Title of the plot.
    """

    # Count the occurrences of each category
    counts = df[variable].value_counts()
    
    plt.figure(figsize=(4, 6))
    plt.pie(counts, labels=legend, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
    plt.title(title_, fontsize=14)
    plt.axis('equal') 
    plt.show()


def plot_correlation_matrix(df, title='Correlation Matrix'):
    """
    Plots a correlation matrix heatmap for the numeric features in the DataFrame.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64', 'Int64'])
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.show() 




def map(df, sample_size=1200, filename="full_dataset_map.html", random_seed=42):
    """
    Saves a full GeoPandas map to an HTML file and displays a sampled version inline.
    """
    
    print(f"Saving full dataset map to {filename}...")
    n = df.explore()
    n.save(filename)
    
    actual_sample = min(sample_size, len(df))
    
    print(f"Rendering inline map with a sample of {actual_sample} points...")
    sample_data = df.sample(actual_sample, random_state=random_seed)
    m = sample_data.explore()
    
    html_content = m.get_root().render()
    encoded = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
    data_uri = f"data:text/html;charset=utf-8;base64,{encoded}"
    
    display(IFrame(src=data_uri, width="100%", height=500))