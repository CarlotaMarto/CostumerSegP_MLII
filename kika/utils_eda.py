"""
utils_eda.py  —  EDA & preprocessing helpers for the Customer Segmentation project.
"""

import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


BLUE = "#1B4F72"

# ── Missing values ────────────────────────────────────────────────────────────

def get_missing_percent(df):
    """Return a sorted DataFrame of missing-value percentages per column."""
    pct = (df.isnull().sum() / len(df) * 100).round(2)
    return (pd.DataFrame({"Column": pct.index, "Missing_Percent": pct.values})
              .sort_values("Missing_Percent", ascending=False))


def get_missing_report(df):
    """Return missing count and percentage for columns that have any missing values."""
    count = df.isnull().sum()
    pct   = (count / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": count, "Percentage (%)": pct})
    return report[report["Missing Count"] > 0].sort_values("Percentage (%)", ascending=False)


def plot_missing_heatmap(df, title="Missing Values Heatmap"):
    """Heatmap showing the location of missing values across the dataset."""
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ── Feature engineering ───────────────────────────────────────────────────────

def get_education_info(row):
    """Extract education level from customer_name prefix.

    Returns a Series of [education_years, clean_name].
    Years encoding: 12 = no degree, 15 = BSc, 17 = MSc, 22 = PhD.
    Using years instead of ordinal 0-3 gives ~10x more variance for
    distance-based clustering.
    """
    if pd.isna(row["customer_name"]):
        return pd.Series([12, ""])
    name = str(row["customer_name"]).strip()
    for level, pattern in [(15, r"^bsc\.\s+"), (17, r"^msc\.\s+"), (22, r"^phd\.\s+")]:
        if re.match(pattern, name, flags=re.IGNORECASE):
            return pd.Series([level, re.sub(pattern, "", name, flags=re.IGNORECASE).strip()])
    return pd.Series([12, name])


def apply_cyclic_transformation(df, col, max_val=24):
    """Replace a cyclic column with its sine and cosine components.

    Ensures that e.g. 23h and 0h are treated as close in distance space,
    which a raw integer would not achieve.
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    temp = pd.to_numeric(df[col], errors="coerce").clip(upper=max_val)
    df[f"{col}_sin"] = np.sin(2 * np.pi * temp / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * temp / max_val)
    return df


# ── Imputation & preprocessing ────────────────────────────────────────────────

def apply_knn_imputation(df, n_neighbors=5, exclude_cols=None):
    """KNN imputation on numeric columns, scaling first so distances are fair.

    Columns in exclude_cols are kept but excluded from the distance calculation
    (useful for clean binary flags like customer_loyalty_flag).
    """
    df_out       = df.copy()
    exclude_cols = [c for c in (exclude_cols or []) if c in df_out.columns]
    num_cols     = [c for c in df_out.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]
    if not num_cols:
        return df_out
    scaler        = StandardScaler()
    scaled        = scaler.fit_transform(df_out[num_cols])
    imputed       = KNNImputer(n_neighbors=n_neighbors).fit_transform(scaled)
    df_out[num_cols] = scaler.inverse_transform(imputed)
    return df_out


def validate_imputation(df_orig, df_imputed, columns):
    """Warn if imputation produced negative values or inflated the max beyond 1.5×."""
    issues = []
    for col in columns:
        if col not in df_imputed.columns or col not in df_orig.columns:
            continue
        if df_imputed[col].min() < 0 and col not in ["longitude", "latitude",
                                                       "typical_hour_sin", "typical_hour_cos"]:
            issues.append(f"{col} has negative values: {df_imputed[col].min():.2f}")
        orig_max = df_orig[col].max()
        if pd.notna(orig_max) and orig_max != 0 and df_imputed[col].max() > orig_max * 1.5:
            issues.append(f"{col} max grew from {orig_max:.2f} to {df_imputed[col].max():.2f}")
    if issues:
        print("Imputation warnings:")
        for i in issues: print(f"  - {i}")
    else:
        print("Imputation validation passed.")
    return len(issues) == 0


# ── Outlier handling ──────────────────────────────────────────────────────────

def cap_bounded_features(df):
    """Cap count columns at domain-sensible ceilings before KNN imputation.

    Prevents extreme counts (e.g. kids_home=8) from pulling KNN neighbours
    toward unrealistic values.
    """
    caps   = {"kids_home": 3, "teens_home": 2, "number_complaints": 2,
               "distinct_stores_visited": 6}
    df_out = df.copy()
    for col, cap in caps.items():
        if col in df_out.columns:
            n = (df_out[col] > cap).sum()
            if n: print(f"  {col}: {n} values capped at {cap}")
            df_out[col] = df_out[col].clip(upper=cap)
    return df_out


def handle_extreme_outliers(df, columns, strategy="cap"):
    """Cap extreme values using the 3.0 × IQR rule (no rows dropped)."""
    df_out = df.copy()
    for col in columns:
        if col not in df_out.columns:
            continue
        q1, q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
        iqr    = q3 - q1
        df_out[col] = df_out[col].clip(lower=q1 - 3*iqr, upper=q3 + 3*iqr)
    return df_out


def cap_spend_outliers(df, iqr_multiplier=1.75):
    """Cap lifetime_spend_* columns before computing share-of-wallet ratios.

    Capping first prevents one extreme spender from distorting their own
    pct_spend_* values and making their mix incomparable with similar customers.
    """
    df_out = df.copy()
    for col in [c for c in df_out.columns if c.startswith("lifetime_spend_")]:
        q1, q3 = df_out[col].dropna().quantile(0.25), df_out[col].dropna().quantile(0.75)
        upper  = q3 + iqr_multiplier * (q3 - q1)
        n      = (df_out[col] > upper).sum()
        if n: print(f"  {col}: {n} values capped at {upper:.0f}")
        df_out[col] = df_out[col].clip(upper=upper)
    return df_out


def remove_semi_constant_features(df, threshold=0.99, exclude_cols=None):
    """Drop columns where one value represents ≥ threshold of all rows."""
    exclude_cols = exclude_cols or []
    to_drop      = [col for col in df.columns if col not in exclude_cols
                    and df[col].value_counts(normalize=True, dropna=False).iloc[0] >= threshold]
    print(f"Semi-constant columns removed (≥{threshold*100:.0f}%): {to_drop}")
    return df.drop(columns=to_drop)


# ── Correlation analysis ──────────────────────────────────────────────────────

def get_high_correlations(df, threshold=0.7):
    """Return pairs of numeric variables with absolute correlation above threshold."""
    corr  = df.select_dtypes(include=[np.number]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [(col, row, upper.loc[row, col])
             for row in upper.index for col in upper.columns
             if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > threshold]
    return (pd.DataFrame(pairs, columns=["Variable 1", "Variable 2", "Correlation"])
              .sort_values("Correlation", ascending=False))


# ── Scaling & export ──────────────────────────────────────────────────────────

def build_clustering_dataset(df, columns_to_drop=None, binary_features=None):
    """Select numeric columns, drop those that shouldn't drive clustering distances."""
    columns_to_drop = columns_to_drop or []
    result = df.select_dtypes(include=[np.number]).copy()
    result = result.drop(columns=[c for c in columns_to_drop if c in result.columns],
                         errors="ignore")
    return result


def scale_clustering_dataset(df, binary_features=None, scaler="robust"):
    """Scale continuous features while leaving binary/ordinal flags untouched.

    The clustering notebook re-scales from the unscaled file, so this function
    is kept here for completeness but the exported unscaled file is the one
    actually consumed by the clustering notebook.
    """
    binary_features = [c for c in (binary_features or []) if c in df.columns]
    scale_cols      = [c for c in df.columns if c not in binary_features]
    scalers         = {"robust": RobustScaler(), "standard": StandardScaler(),
                       "minmax": MinMaxScaler()}
    if scaler not in scalers:
        raise ValueError(f"scaler must be one of {list(scalers)}")
    scaled           = scalers[scaler].fit_transform(df[scale_cols])
    scaled_df        = pd.DataFrame(scaled, columns=scale_cols, index=df.index)
    for c in binary_features:
        scaled_df[c] = df[c].values
    return scaled_df[df.columns]


def validate_customer_coverage(df_clean, df_original, id_col="customer_id"):
    """Assert that no customer was lost during preprocessing."""
    orig_ids  = set(df_original[id_col].astype(int))
    clean_ids = (set(df_clean.index.astype(int)) if df_clean.index.name == id_col
                 else set(df_clean[id_col].astype(int)))
    dropped   = orig_ids - clean_ids
    print(f"Original: {len(orig_ids):,}  |  Remaining: {len(clean_ids):,}  |  Lost: {len(dropped):,}")
    return dropped


def export_clustering_data(scaled_df, unscaled_df=None, output_dir="../datasets",
                           scaled_name="info_clustering_ready.csv",
                           unscaled_name="info_clustering_unscaled.csv"):
    """Export scaled (for reference) and unscaled (for clustering) datasets."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    scaled_df.to_csv(os.path.join(output_dir, scaled_name), index=True)
    print(f"Scaled   → {output_dir}/{scaled_name}")
    if unscaled_df is not None:
        unscaled_df.to_csv(os.path.join(output_dir, unscaled_name), index=True)
        print(f"Unscaled → {output_dir}/{unscaled_name}")


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_numeric_distributions(df, columns=None, ncols=3, color=BLUE):
    """Histogram + KDE for a set of numeric columns."""
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    columns = [c for c in columns if c in df.columns]
    nrows   = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), kde=True, color=color, ax=ax)
        ax.set_title(col); ax.set_xlabel("")
    for ax in axes[len(columns):]: ax.axis("off")
    plt.tight_layout(); plt.show()


def plot_boxplots(df, columns, ncols=3, color=BLUE, title_suffix=""):
    """Boxplots to inspect outliers across several columns."""
    columns = [c for c in columns if c in df.columns]
    nrows   = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, columns):
        sns.boxplot(x=df[col].dropna(), color=color, ax=ax)
        ax.set_title(f"{col}{title_suffix}"); ax.set_xlabel("")
    for ax in axes[len(columns):]: ax.axis("off")
    plt.tight_layout(); plt.show()


def plot_spend_breakdown(df, pct_cols=None, color=BLUE):
    """Average share-of-wallet per product category."""
    pct_cols = pct_cols or [c for c in df.columns if c.startswith("pct_spend_")]
    means    = df[pct_cols].mean().sort_values(ascending=True)
    labels   = [c.replace("pct_spend_", "").capitalize() for c in means.index]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, means.values, color=color)
    plt.title("Average Share of Wallet by Category")
    plt.xlabel("Average share of total spend")
    plt.tight_layout(); plt.show()


def plot_categorical_counts(df, columns, ncols=3, color=BLUE):
    """Count plots for low-cardinality / binary columns."""
    columns = [c for c in columns if c in df.columns]
    nrows   = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, columns):
        sns.countplot(x=df[col], order=df[col].value_counts().index, color=color, ax=ax)
        ax.set_title(col); ax.set_xlabel("")
    for ax in axes[len(columns):]: ax.axis("off")
    plt.tight_layout(); plt.show()


def cor_heatmap(corr_matrix, color=BLUE):
    """Triangular correlation heatmap."""
    plt.figure(figsize=(20, 15))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap=sns.light_palette(color, as_cmap=True),
                center=0, square=True, linewidths=0.5,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.5})
    plt.title("Correlation Heatmap", fontsize=20, fontweight="bold", pad=25)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()
