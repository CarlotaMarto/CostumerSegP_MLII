import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# MISSING VALUES AND DATA INTEGRITY
# ============================================================

def get_missing_percent(df):
    """
    Returns a DataFrame with the percentage of missing values per column.
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
    Returns a DataFrame with the count and percentage of missing values.
    Only columns with missing values are displayed.
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
    Calculates the percentage of rows that meet a specific condition.
    """
    return round((condition.sum() / len(df)) * 100, 2)


def get_invalid_years(df, year_col="year_first_transaction"):
    """
    Identifies rows where the transaction year is greater than the current year.
    """
    current_year = datetime.now().year

    if year_col not in df.columns:
        raise ValueError(f"Column '{year_col}' was not found in the DataFrame.")

    invalid_rows = df[df[year_col] > current_year]

    return invalid_rows


# ============================================================
# DATA TRANSFORMATION FUNCTIONS
# ============================================================

def get_education_info(row):
    """
    Extracts the education level from the customer name and returns:
    - education level:
        0 = No degree identified
        1 = BSc
        2 = MSc
        3 = PhD
    - cleaned customer name
    """
    name = str(row["customer_name"])
    lower_name = name.lower()

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

    else:
        education_level = 0
        clean_name = name.strip()

    return pd.Series([education_level, clean_name])


def apply_cyclic_transformation(df, col, max_val=24):
    """
    Applies cyclic transformation to a numerical column.

    This is useful for variables such as hour, day of week or month,
    where the highest value is close to the lowest value in a cycle.

    Creates two new columns:
    - column_sin
    - column_cos
    """
    df_transformed = df.copy()

    if col not in df_transformed.columns:
        raise ValueError(f"Column '{col}' was not found in the DataFrame.")

    temp_col = pd.to_numeric(df_transformed[col], errors="coerce")

    df_transformed[f"{col}_sin"] = np.sin(2 * np.pi * temp_col / max_val)
    df_transformed[f"{col}_cos"] = np.cos(2 * np.pi * temp_col / max_val)

    return df_transformed


# ============================================================
# ADVANCED PREPROCESSING
# ============================================================

def apply_knn_imputation(df, n_neighbors=5):
    """
    Applies KNN imputation to numerical columns using StandardScaler first.

    Why scaling is needed:
    KNNImputer uses distance between rows. Without scaling, variables with
    large magnitudes, such as spending columns, dominate the distance calculation.

    Non-numerical columns remain unchanged.
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


def handle_extreme_outliers(df, columns, strategy="cap"):
    """
    Handles extreme outliers using the 3.0 * IQR rule.

    Extreme outliers are values outside:
    [Q1 - 3 * IQR, Q3 + 3 * IQR]

    Parameters:
    - df: input DataFrame
    - columns: list of numerical columns to process
    - strategy: currently supports 'cap'

    Returns:
    - DataFrame with treated outliers
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
                np.where(
                    df_out[col] < lower_bound,
                    lower_bound,
                    df_out[col]
                )
            )

        else:
            raise ValueError("Invalid strategy. Currently, only 'cap' is supported.")

    return df_out


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def get_high_correlations(df, threshold=0.7):
    """
    Identifies pairs of numerical variables with absolute correlation
    above the selected threshold.
    """
    corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    high_correlations = [
        (column, row, upper_triangle.loc[row, column])
        for row in upper_triangle.index
        for column in upper_triangle.columns
        if pd.notna(upper_triangle.loc[row, column])
        and upper_triangle.loc[row, column] > threshold
    ]

    result = pd.DataFrame(
        high_correlations,
        columns=["Variable 1", "Variable 2", "Correlation"]
    )

    return result.sort_values(by="Correlation", ascending=False)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_missing_heatmap(df, title="Missing Values Heatmap"):
    """
    Plots a heatmap to visualize the location of missing values in the DataFrame.
    """
    plt.figure(figsize=(10, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        yticklabels=False,
        cmap="viridis"
    )

    plt.title(title)
    plt.show()


def cor_heatmap(corr_matrix, color="#1B4F72"):
    """
    Plots a correlation heatmap using a triangular mask.

    This function keeps the original name 'cor_heatmap'
    so it works with the existing notebook code.
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
    Calls cor_heatmap() internally.
    """
    return cor_heatmap(corr_matrix, color=color)