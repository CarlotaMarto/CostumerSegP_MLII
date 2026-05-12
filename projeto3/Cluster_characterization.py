import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def read_cluster_data(filepath="dataset_cluster_final.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'dataset_cluster_final.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    
def read_makro(filepath="makro_cluster.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'makro_cluster.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    
def read_info(filepath="customer_info.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'makro_cluster.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def plot_all_numeric_means_by_cluster(df, cluster_col='kmeans_cluster', binary_cols=None):
    """
    Plots the mean of each numeric feature across clusters.
    One bar chart per feature, excluding binary columns.

    Assumes all columns are numeric.
    """
    binary_cols = binary_cols or []

    feature_cols = [col for col in df.columns if col not in binary_cols + [cluster_col]]

    cluster_means = df.groupby(cluster_col)[feature_cols].mean().reset_index()

    df_melted = cluster_means.melt(
        id_vars=cluster_col,
        var_name='Feature',
        value_name='Average'
    )

    # Plot each feature
    for feature in df_melted['Feature'].unique():
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=df_melted[df_melted['Feature'] == feature],
            x=cluster_col,
            y='Average',
            hue=cluster_col,
            palette='pastel',
            legend=False
        )
        plt.title(f'Average {feature} per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(f'{feature}')
        plt.tight_layout()
        plt.show()

def summarize_each_feature_by_cluster(df, cluster_col='Cluster', binary_cols=None):
    """
    For each numeric feature, create a separate summary table with mean, min, max per cluster.
    
    Returns a dict: {feature_name: DataFrame_summary}
    """
    binary_cols = binary_cols or []

    feature_cols = [col for col in df.columns if col not in binary_cols + [cluster_col] and pd.api.types.is_numeric_dtype(df[col])]

    summaries = {}

    for feature in feature_cols:
        summary = df.groupby(cluster_col)[feature].agg(['mean', 'min', 'max']).reset_index()
        summaries[feature] = summary

        print(f"\n--- Summary for feature: {feature} ---")
        display(summary)

    return summaries


cluster_mapping = {
    0: 'Vegetarian',
    1: "Wellness Urbanites",
    2: 'Student',
    3: "Extended Household",
    4: "Tech Entusiast",
    5: "Mature Independent"
}

def map_cluster_label(label):
    if isinstance(label, int) or (isinstance(label, str) and label.isdigit()):
        return cluster_mapping.get(int(label), label)
    else:
        return label