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

import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# ============================================================
# Missing values and data integrity
# ============================================================


def get_missing_percent(df):
    """Return the percentage of missing values for each column."""
    missing_percent = (df.isnull().sum() / len(df)) * 100

    report = pd.DataFrame(
        {"Column": missing_percent.index, "Missing_Percent": missing_percent.values}
    )

    report["Missing_Percent"] = report["Missing_Percent"].round(2)
    report = report.sort_values("Missing_Percent", ascending=False)

    return report



def get_missing_report(df):
    """Return a missing values report with count and percentage."""
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    report = pd.DataFrame(
        {"Missing Count": missing_count, "Percentage (%)": missing_percentage.round(2)}
    )

    report = report[report["Missing Count"] > 0]
    report = report.sort_values(by="Percentage (%)", ascending=False)

    return report



def calculate_percentage(df, condition):
    """Calculate the percentage of rows that satisfy a given condition."""
    return round((condition.sum() / len(df)) * 100, 2)



def get_invalid_years(df, year_col="year_first_transaction"):
    """Identify rows where the transaction year is greater than the current year."""
    current_year = datetime.now().year

    if year_col not in df.columns:
        raise ValueError(f"Column '{year_col}' was not found in the DataFrame.")

    return df[df[year_col] > current_year]


# ============================================================
# Feature engineering and data transformation
# ============================================================


def get_education_info(row):
    """Extract education level safely from the beginning of customer_name.

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

    name = str(row["customer_name"]).strip()
    patterns = {
        1: r"^bsc\.\s+",
        2: r"^msc\.\s+",
        3: r"^phd\.\s+",
    }

    for level, pattern in patterns.items():
        if re.match(pattern, name, flags=re.IGNORECASE):
            clean_name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
            return pd.Series([level, clean_name])

    return pd.Series([0, name])



def apply_cyclic_transformation(df, col, max_val=24):
    """Apply cyclic transformation to a numerical cyclic column."""
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


def apply_knn_imputation(df, n_neighbors=5, exclude_cols=None):
    """Apply KNN imputation to numerical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    n_neighbors : int, default=5
        Number of neighbors used by KNNImputer.
    exclude_cols : list, optional
        Columns excluded from the distance calculation and imputation.
        Useful for identifiers and already-clean binary flags such as customer_id.

    Notes
    -----
    Numerical features are scaled before imputation because KNNImputer uses
    distances between rows.
    """
    df_imputed = df.copy()
    exclude_cols = exclude_cols or []
    exclude_cols = [col for col in exclude_cols if col in df_imputed.columns]

    numeric_cols = [
        col
        for col in df_imputed.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

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
        index=df_imputed.index,
    )

    return df_imputed



def validate_imputation(df_original, df_imputed, columns):
    """Validate whether imputation created unrealistic values."""
    issues = []

    for col in columns:
        if col not in df_imputed.columns or col not in df_original.columns:
            continue

        if df_imputed[col].min() < 0 and col not in ["longitude", "latitude"]:
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
    """Handle extreme outliers using the 3.0 * IQR rule."""
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
                np.where(df_out[col] < lower_bound, lower_bound, df_out[col]),
            )
        else:
            raise ValueError("Invalid strategy. Currently, only 'cap' is supported.")

    return df_out



def remove_semi_constant_features(df, threshold=0.99, exclude_cols=None):
    """Remove columns where one value represents at least `threshold` of rows."""
    exclude_cols = exclude_cols or []
    semi_constant_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        value_counts = df[col].value_counts(normalize=True, dropna=False)

        if len(value_counts) == 0:
            continue

        most_frequent_ratio = value_counts.iloc[0]

        if most_frequent_ratio >= threshold:
            semi_constant_cols.append(col)

    print(
        f"Semi-constant columns removed (>={threshold * 100:.0f}% identical): "
        f"{semi_constant_cols}"
    )

    return df.drop(columns=semi_constant_cols)


# ============================================================
# Scaling and clustering helper functions
# ============================================================


def scale_robust_excluding_binary(df, binary_cols=None):
    """Apply RobustScaler only to continuous numerical columns."""
    binary_cols = binary_cols or []
    binary_cols = [col for col in binary_cols if col in df.columns]

    scale_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in binary_cols
    ]

    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df[scale_cols])

    scaled_df = pd.DataFrame(scaled_array, columns=scale_cols, index=df.index)

    return pd.concat([scaled_df, df[binary_cols]], axis=1)



def test_scalers_kmeans(df, binary_cols=None, k=4, random_state=42):
    """Compare StandardScaler, MinMaxScaler and RobustScaler using KMeans."""
    binary_cols = binary_cols or []
    binary_cols = [col for col in binary_cols if col in df.columns]

    X = df.drop(columns=binary_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        raise ValueError("No numerical columns available for scaling and clustering.")

    scalers = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(),
        "Robust": RobustScaler(),
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

    scaled_df = pd.DataFrame(X_final_scaled, columns=X.columns, index=X.index)

    for b_col in binary_cols:
        scaled_df[b_col] = df[b_col].values

    return best_scaler_name, best_score, scaled_df


# ============================================================
# Correlation analysis
# ============================================================


def get_high_correlations(df, threshold=0.7):
    """Identify pairs of numerical variables with absolute correlation above threshold."""
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
        high_correlations, columns=["Variable 1", "Variable 2", "Correlation"]
    )

    return result.sort_values(by="Correlation", ascending=False)


# ============================================================
# Visualization functions
# ============================================================


def plot_missing_heatmap(df, title="Missing Values Heatmap"):
    """Plot a heatmap showing the location of missing values in the dataset."""
    plt.figure(figsize=(10, 5))

    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")

    plt.title(title)
    plt.tight_layout()
    plt.show()



def cor_heatmap(corr_matrix, color="#1B4F72"):
    """Plot a triangular correlation heatmap."""
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
        cbar_kws={"shrink": 0.5},
    )

    plt.title("Correlation Heatmap", fontsize=20, fontweight="bold", pad=25)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_correlation_heatmap(corr_matrix, color="#1B4F72"):
    """Alternative function name for the correlation heatmap."""
    return cor_heatmap(corr_matrix, color=color)
