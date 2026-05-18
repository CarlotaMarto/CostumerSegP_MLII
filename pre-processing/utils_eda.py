"""
utils_eda.py

Utility functions for the Customer Segmentation project.
This file supports the EDA and preprocessing workflow by grouping together
reusable functions for missing values, feature engineering, outlier handling,
scaling, correlations and visualizations.
"""

# ============================================================
# Imports
# ============================================================

from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# ============================================================
# Missing values and data integrity
# ============================================================

def get_missing_percent(df):
    """
    Return the percentage of missing values for each column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame with column names and missing percentages.
    """
    missing_percent = (df.isnull().sum() / len(df)) * 100

    report = pd.DataFrame({
        "Column": missing_percent.index,
        "Missing_Percent": missing_percent.values
    })

    report["Missing_Percent"] = report["Missing_Percent"].round(2)
    report = report.sort_values("Missing_Percent", ascending=False)

    return report


def get_missing_report(df):
    """
    Return a missing values report with count and percentage.
    Only columns with missing values are shown.
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "Missing Count": missing_count,
        "Percentage (%)": missing_percentage.round(2)
    })

    report = report[report["Missing Count"] > 0]
    report = report.sort_values(by="Percentage (%)", ascending=False)

    return report


def calculate_percentage(df, condition):
    """
    Calculate the percentage of rows that satisfy a given condition.
    """
    return round((condition.sum() / len(df)) * 100, 2)


def get_invalid_years(df, year_col="year_first_transaction"):
    """
    Identify rows where the transaction year is greater than the current year.
    """
    current_year = datetime.now().year

    if year_col not in df.columns:
        raise ValueError(f"Column '{year_col}' was not found in the DataFrame.")

    return df[df[year_col] > current_year]


# ============================================================
# Feature engineering and data transformation
# ============================================================

def get_education_info(row):
    """
    Extract the education level from the customer name.

    Returns
    -------
    pandas.Series
        [education_level, clean_customer_name]

    Education encoding:
        0 = No degree identified
        1 = BSc
        2 = MSc
        3 = PhD
    """
    if pd.isna(row["customer_name"]):
        return pd.Series([0, ""])

    name = str(row["customer_name"])
    lower_name = name.lower()

    education_level = 0
    clean_name = name.strip()

    if "bsc." in lower_name:
        education_level = 1
        clean_name = (
            name.replace("Bsc.", "")
                .replace("bsc.", "")
                .replace("BSC.", "")
                .strip()
        )

    elif "msc." in lower_name:
        education_level = 2
        clean_name = (
            name.replace("Msc.", "")
                .replace("msc.", "")
                .replace("MSC.", "")
                .strip()
        )

    elif "phd." in lower_name:
        education_level = 3
        clean_name = (
            name.replace("Phd.", "")
                .replace("phd.", "")
                .replace("PHD.", "")
                .strip()
        )

    return pd.Series([education_level, clean_name])


def apply_cyclic_transformation(df, col, max_val=24):
    """
    Apply cyclic transformation to a numerical column.

    This is useful for variables such as hour, day of week or month,
    where the highest value is close to the lowest value in a cycle.

    Creates two new columns:
        - column_sin
        - column_cos
    """
    df_transformed = df.copy()

    if col not in df_transformed.columns:
        raise ValueError(f"Column '{col}' was not found in the DataFrame.")

    if df_transformed[col].isna().all():
        print(f"Column '{col}' is all NaN. Cyclic transformation skipped.")
        df_transformed[f"{col}_sin"] = np.nan
        df_transformed[f"{col}_cos"] = np.nan
        return df_transformed

    temp_col = pd.to_numeric(df_transformed[col], errors="coerce")

    if temp_col.max() > max_val:
        print(f"Column '{col}' has values greater than {max_val}. Values were clipped.")
        temp_col = temp_col.clip(upper=max_val)

    df_transformed[f"{col}_sin"] = np.sin(2 * np.pi * temp_col / max_val)
    df_transformed[f"{col}_cos"] = np.cos(2 * np.pi * temp_col / max_val)

    return df_transformed


# ============================================================
# Imputation and preprocessing
# ============================================================

def apply_knn_imputation(df, n_neighbors=5):
    """
    Apply KNN imputation to numerical columns.

    Numerical features are scaled before imputation because KNNImputer
    uses distances between rows. Without scaling, features with larger
    magnitudes could dominate the imputation process.

    Non-numerical columns are kept unchanged.
    """
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return df_imputed

    numeric_df = df_imputed[numeric_cols]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled_data = imputer.fit_transform(scaled_data)

    imputed_data = scaler.inverse_transform(imputed_scaled_data)

    df_imputed[numeric_cols] = pd.DataFrame(
        imputed_data,
        columns=numeric_cols,
        index=df_imputed.index
    )

    return df_imputed


def validate_imputation(df_original, df_imputed, columns):
    """
    Validate whether imputation created unrealistic values.

    This function checks for negative values in columns that should not
    be negative and for extreme inflation in maximum values.
    """
    issues = []

    for col in columns:
        if col not in df_imputed.columns or col not in df_original.columns:
            continue

        if df_imputed[col].min() < 0 and col != "longitude":
            issues.append(
                f"{col} has negative values after imputation: {df_imputed[col].min():.2f}"
            )

        original_max = df_original[col].max()
        imputed_max = df_imputed[col].max()

        if pd.notna(original_max) and original_max != 0:
            if imputed_max > original_max * 1.5:
                issues.append(
                    f"{col} max increased from {original_max:.2f} to {imputed_max:.2f}"
                )

    if issues:
        print("Imputation validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Imputation validation passed.")

    return len(issues) == 0


# ============================================================
# Outlier handling and feature selection
# ============================================================

def handle_extreme_outliers(df, columns, strategy="cap"):
    """
    Handle extreme outliers using the 3.0 * IQR rule.

    Extreme outliers are values outside:
        [Q1 - 3 * IQR, Q3 + 3 * IQR]

    Currently supported strategy:
        - cap: values below/above the limits are capped to the limits.
    """
    df_out = df.copy()

    for col in columns:
        if col not in df_out.columns:
            continue

        q1 = df_out[col].quantile(0.25)
        q3 = df_out[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (3.0 * iqr)
        upper_bound = q3 + (3.0 * iqr)

        if strategy == "cap":
            df_out[col] = np.where(
                df_out[col] > upper_bound,
                upper_bound,
                np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            )
        else:
            raise ValueError("Invalid strategy. Currently, only 'cap' is supported.")

    return df_out


def remove_semi_constant_features(df, threshold=0.99):
    """
    Remove columns where one value represents more than the selected threshold.

    These columns have very low variance and usually do not contribute much
    to clustering or distance-based models.
    """
    semi_constant_cols = []

    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False)

        if len(value_counts) == 0:
            continue

        most_frequent_ratio = value_counts.iloc[0]

        if most_frequent_ratio >= threshold:
            semi_constant_cols.append(col)

    print(f"Semi-constant columns removed (>{threshold * 100:.0f}% identical): {semi_constant_cols}")

    return df.drop(columns=semi_constant_cols)


# ============================================================
# Scaling and clustering helper functions
# ============================================================

def scale_robust_excluding_binary(df, binary_cols=None):
    """
    Apply RobustScaler only to continuous numerical columns.

    Binary columns are kept unchanged because scaling 0/1 indicators
    can make their interpretation less direct.
    """
    binary_cols = binary_cols or []
    binary_cols = [col for col in binary_cols if col in df.columns]

    scale_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in binary_cols
    ]

    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df[scale_cols])

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=scale_cols,
        index=df.index
    )

    return pd.concat([scaled_df, df[binary_cols]], axis=1)


def test_scalers_kmeans(df, binary_cols=None, k=4, random_state=42):
    """
    Compare StandardScaler, MinMaxScaler and RobustScaler using KMeans.

    The scaler with the highest silhouette score is selected.
    Binary columns can be excluded from scaling and added back afterwards.

    Returns
    -------
    tuple
        best_scaler_name, best_score, scaled_df
    """
    binary_cols = binary_cols or []
    binary_cols = [col for col in binary_cols if col in df.columns]

    X = df.drop(columns=binary_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numerical columns available for scaling and clustering.")

    scalers = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(),
        "Robust": RobustScaler()
    }

    scaler_results = {}

    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        scaler_results[name] = score

        print(f"{name} Scaler - Silhouette Score: {score:.4f}")

    best_scaler_name = max(scaler_results, key=scaler_results.get)
    best_score = scaler_results[best_scaler_name]

    print(f"\nBest Scaler: {best_scaler_name} (Silhouette Score: {best_score:.4f})")

    final_scaler = scalers[best_scaler_name]
    X_final_scaled = final_scaler.fit_transform(X)

    scaled_df = pd.DataFrame(
        X_final_scaled,
        columns=X.columns,
        index=X.index
    )

    if binary_cols:
        scaled_df[binary_cols] = df[binary_cols].values

    return best_scaler_name, best_score, scaled_df


# ============================================================
# Correlation analysis
# ============================================================

def get_high_correlations(df, threshold=0.7):
    """
    Identify pairs of numerical variables with absolute correlation
    above the selected threshold.
    """
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_correlations = []

    for row in upper_triangle.index:
        for column in upper_triangle.columns:
            corr_value = upper_triangle.loc[row, column]

            if pd.notna(corr_value) and corr_value > threshold:
                high_correlations.append((column, row, corr_value))

    result = pd.DataFrame(
        high_correlations,
        columns=["Variable 1", "Variable 2", "Correlation"]
    )

    return result.sort_values(by="Correlation", ascending=False)


# ============================================================
# Visualization functions
# ============================================================

def plot_missing_heatmap(df, title="Missing Values Heatmap"):
    """
    Plot a heatmap showing the location of missing values in the dataset.
    """
    plt.figure(figsize=(10, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        yticklabels=False,
        cmap="viridis"
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def cor_heatmap(corr_matrix, color="#1B4F72"):
    """
    Plot a triangular correlation heatmap.

    The function keeps the original name 'cor_heatmap' so it remains
    compatible with the notebook.
    """
    plt.figure(figsize=(20, 15))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.light_palette(color, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.5}
    )

    plt.title(
        "Correlation Heatmap",
        fontsize=20,
        fontweight="bold",
        pad=25
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix, color="#1B4F72"):
    """
    Alternative function name for the correlation heatmap.
    """
    return cor_heatmap(corr_matrix, color=color)
