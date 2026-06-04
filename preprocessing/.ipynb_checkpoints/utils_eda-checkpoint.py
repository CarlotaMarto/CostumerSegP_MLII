"""Utility functions for the EDA and preprocessing workflow.

The notebook keeps the analysis narrative. This module contains reusable data
quality checks, feature engineering, outlier handling and diagnostic plots.
"""


import os
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


def get_invalid_years(df, year_col="year_first_transaction"):
    """Identify rows where the transaction year is greater than the current year.

    Note
    ----
    Use this to FLAG invalid years and set them to NaN (then impute), rather
    than dropping the rows. Dropping rows here silently loses customers and
    breaks the "every customer ends up in a segment" guarantee.
    """
    current_year = datetime.now().year

    if year_col not in df.columns:
        raise ValueError(f"Column '{year_col}' was not found in the DataFrame.")

    return df[df[year_col] > current_year]


def get_education_info(row):
    """Extract education level safely from the beginning of customer_name.

    Returns
    -------
    pandas.Series
        [education_level, clean_customer_name]

    Education encoding (YEARS of education = interval scale):
        12 = Unknown / High-School
        15 = BSc
        17 = MSc
        22 = PhD

    Using years rather than 0/1/2/3 codes gives a genuine interval scale: the
    real gaps between degrees (e.g. 5 years from MSc to PhD vs 2 from BSc to
    MSc) are preserved, which matters for the Euclidean distance in K-Means.
    Being a true numeric scale, it should be scaled as a CONTINUOUS feature
    (with the same scaler as the others), not kept apart as an ordinal code.
    The chosen year values are a reasonable assumption about typical degree
    duration, not a measured fact, and should be documented as such.
    """
    if pd.isna(row["customer_name"]):
        return pd.Series([12, ""])

    name = str(row["customer_name"]).strip()
    patterns = {
        15: r"^bsc\.\s+",
        17: r"^msc\.\s+",
        22: r"^phd\.\s+",
    }

    for years, pattern in patterns.items():
        if re.match(pattern, name, flags=re.IGNORECASE):
            clean_name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()
            return pd.Series([years, clean_name])

    return pd.Series([12, name])


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


def apply_knn_imputation(df, n_neighbors=5, exclude_cols=None):
    """Apply KNN imputation to numerical columns.

    Parameters
    ----------
    df : pandas.DataFrame
    n_neighbors : int, default=5
    exclude_cols : list, optional
        Columns excluded from the distance calculation and imputation
        (identifiers, binary flags such as is_male / customer_loyalty_flag,
        and the ordinal education_level so its codes are not averaged).

    Notes
    -----
    Numerical features are scaled before imputation because KNNImputer uses
    distances between rows. StandardScaler ignores NaN during fit, so the
    presence of missing values is fine.

    ORDER: run this AFTER gross outliers have been set aside
    (`split_outliers_iqr`). Imputing with outliers still in the data
    distorts both the scaler statistics and the neighbour distances.
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
    """Validate whether imputation created unrealistic values.

    Fix: cyclic features (typical_hour_sin / typical_hour_cos) and geographic
    coordinates are LEGITIMATELY negative, so they are not flagged.
    """
    issues = []
    allowed_negative = {
        "longitude",
        "latitude",
        "typical_hour_sin",
        "typical_hour_cos",
    }

    for col in columns:
        if col not in df_imputed.columns or col not in df_original.columns:
            continue

        if df_imputed[col].min() < 0 and col not in allowed_negative:
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


def split_outliers_iqr(
    df,
    feature_cols,
    iqr_multiplier=3.0,
    min_flagged_features=1,
    max_remove_pct=5.0,
):
    """Detect and SEPARATE outliers on raw, skewed features (do not delete them).

    This mirrors the report's logic: outliers are kept aside in a separate
    dataset instead of being discarded, so they can later be assigned to the
    nearest cluster centroid. Run it on the RAW absolute `lifetime_spend_*`
    columns plus other tailed numerics, BEFORE feature engineering.

    Parameters
    ----------
    df : pandas.DataFrame
    feature_cols : list
        Continuous columns to inspect (raw spends + tailed behavioural vars).
        Exclude binary flags, ordinal codes and geo coordinates.
    iqr_multiplier : float, default=3.0
        Bounds are Q1 - m*IQR and Q3 + m*IQR. Higher = more conservative.
    min_flagged_features : int, default=1
        A row is an outlier when it is extreme on at least this many columns.
        Increase it (e.g. 2 or 3) for a stricter "consensus across features".
    max_remove_pct : float, default=5.0
        Safeguard. If more than this share would be removed, a warning is
        printed but the split is still returned for review.

    Returns
    -------
    (regular_df, outlier_df, summary_df)
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise ValueError("None of the requested feature_cols are in the DataFrame.")

    flags = pd.DataFrame(index=df.index)
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - iqr_multiplier * iqr
        high = q3 + iqr_multiplier * iqr
        flags[col] = (values < low) | (values > high)

    n_flagged = flags.sum(axis=1)
    outlier_mask = n_flagged >= min_flagged_features
    outlier_pct = outlier_mask.mean() * 100

    summary = pd.DataFrame(
        {
            "Column": feature_cols,
            "Flagged": [int(flags[c].sum()) for c in feature_cols],
            "Pct": [round(flags[c].mean() * 100, 2) for c in feature_cols],
        }
    ).sort_values("Flagged", ascending=False)

    print(summary.to_string(index=False))
    print(f"\nRows flagged as outliers: {int(outlier_mask.sum()):,} "
          f"({outlier_pct:.2f}%)")

    if outlier_pct > max_remove_pct:
        print(f"Note: this is above the {max_remove_pct:.1f}% reference limit. "
              f"Consider raising iqr_multiplier or min_flagged_features.")

    regular_df = df.loc[~outlier_mask].copy()
    outlier_df = df.loc[outlier_mask].copy()

    print(f"Regular customers: {len(regular_df):,}")
    print(f"Outliers kept aside: {len(outlier_df):,}")
    print(f"Total preserved: {len(regular_df) + len(outlier_df):,}")

    return regular_df, outlier_df, summary


def engineer_clustering_features(
    df,
    current_year,
    spend_prefix="lifetime_spend_",
    keep_absolute_spend=True,
):
    """Build the engineered features used downstream.

    This version deliberately keeps the absolute lifetime spend columns by
    default. Absolute spend variables contain customer value and purchasing
    intensity information, which are useful for distance-based clustering after
    scaling.

    Applies, when the source columns exist:
      * tenure = current_year - year_first_transaction
      * total_children = kids_home + teens_home
      * lifetime_spend_technology = electronics + videogames (originals dropped)
      * log_total_spend = log1p(total_spend)

    Parameters
    ----------
    keep_absolute_spend : bool, default True
        If True, keeps the lifetime_spend_* columns in the exported dataset.
        These columns must be scaled later in the clustering notebook before
        distance-based algorithms are fitted. If False, absolute spend columns
        are dropped after creating the requested aggregate features.

    Notes
    -----
    The helper column total_spend is dropped to avoid keeping a perfect duplicate
    of the spend block. The log_total_spend feature remains as a compact value
    indicator for profiling and optional modelling.
    """
    out = df.copy()

    if "year_first_transaction" in out.columns:
        out["tenure"] = current_year - out["year_first_transaction"]

    if {"kids_home", "teens_home"}.issubset(out.columns):
        out["total_children"] = out["kids_home"] + out["teens_home"]

    if {"lifetime_spend_electronics", "lifetime_spend_videogames"}.issubset(out.columns):
        out["lifetime_spend_technology"] = (
            out["lifetime_spend_electronics"] + out["lifetime_spend_videogames"]
        )
        out = out.drop(columns=["lifetime_spend_electronics", "lifetime_spend_videogames"])

    spend = [c for c in out.columns if c.startswith(spend_prefix)]
    if spend:
        out["total_spend"] = out[spend].sum(axis=1)

        out["log_total_spend"] = np.log1p(np.clip(out["total_spend"], 0, None))

        # Keep absolute spend variables for the scaled clustering alternative,
        # but remove the helper total_spend to avoid a redundant total column.
        drop_cols = ["total_spend"]
        if not keep_absolute_spend:
            drop_cols = spend + drop_cols
        out = out.drop(columns=drop_cols)

    return out


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


def build_clustering_features(df, exclude_cols=None):
    """Return the numeric columns that should DRIVE the clustering distance.

    `exclude_cols` are identity / geographic features that must be kept for
    profiling but must NOT influence the distance, e.g.
        ["is_male", "customer_loyalty_flag", "latitude", "longitude",
         "loyalty_card_number", "year_first_transaction"]
    """
    exclude_cols = set(exclude_cols or [])
    cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    return df[cols].copy()


def _scale_matrix(df, scaler, binary_cols=None, ordinal_cols=None):
    """Scale continuous columns with `scaler`, standardise ordinals, keep binaries as 0/1."""
    binary_cols = [c for c in (binary_cols or []) if c in df.columns]
    ordinal_cols = [c for c in (ordinal_cols or []) if c in df.columns]
    cont_cols = [c for c in df.columns if c not in binary_cols + ordinal_cols]

    parts, names = [], []
    if cont_cols:
        parts.append(scaler.fit_transform(df[cont_cols].astype(float)))
        names += cont_cols
    if ordinal_cols:
        parts.append(StandardScaler().fit_transform(df[ordinal_cols].astype(float)))
        names += ordinal_cols
    if binary_cols:
        parts.append(df[binary_cols].to_numpy(dtype=float))
        names += binary_cols

    X = np.concatenate(parts, axis=1) if parts else df.to_numpy(dtype=float)
    return X, names


def test_scalers_kmeans(
    df,
    exclude_cols=None,
    binary_cols=None,
    ordinal_cols=None,
    k_values=(4, 5, 6, 7, 8),
    random_state=42,
):
    """Compare Standard, MinMax and Robust scalers with KMeans.

    Changes vs the old version:
      * `exclude_cols` are DROPPED from the distance (identity/geo features),
        not merely left unscaled. They no longer influence the clustering.
      * the scaler is chosen by the MEAN silhouette over a range of k, not a
        single hard-coded k=4 (which always favours the smallest k).
      * binary columns are kept as 0/1; ordinal columns are standardised.

    Returns
    -------
    (best_scaler_name, scores_table, scaled_df_with_best_scaler)
        scaled_df_with_best_scaler contains ONLY the clustering features
        (excluded identity/geo columns are not in it).
    """
    features = build_clustering_features(df, exclude_cols=exclude_cols)
    if features.shape[1] == 0:
        raise ValueError("No numerical columns available for clustering.")

    scalers = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(),
        "Robust": RobustScaler(),
    }

    rows = []
    per_scaler_mean = {}
    for name, scaler in scalers.items():
        X, cols = _scale_matrix(features, scaler, binary_cols, ordinal_cols)
        sils = []
        for k in k_values:
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
            s = silhouette_score(X, labels, sample_size=min(8000, len(labels)),
                                 random_state=random_state)
            sils.append(s)
            rows.append({"Scaler": name, "k": k, "Silhouette": round(float(s), 4)})
        per_scaler_mean[name] = float(np.mean(sils))
        print(f"{name:8s} mean silhouette over k={tuple(k_values)}: {per_scaler_mean[name]:.4f}")

    best = max(per_scaler_mean, key=per_scaler_mean.get)
    print(f"\nBest scaler (mean over k): {best} ({per_scaler_mean[best]:.4f})")

    X_best, cols = _scale_matrix(features, scalers[best], binary_cols, ordinal_cols)
    scaled_df = pd.DataFrame(X_best, columns=cols, index=features.index)
    scores_table = (pd.DataFrame(rows)
                    .sort_values(["Scaler", "k"]).reset_index(drop=True))
    return best, scores_table, scaled_df


def set_plot_style(figsize=(10, 6)):
    """Apply the visual style used across the preprocessing notebook."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = figsize


def get_column_groups(df):
    """Return numerical and categorical column names."""
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical_cols, categorical_cols


def plot_missing_percent(missing_df):
    """Plot missing-value percentages by column."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Missing_Percent", y="Column", data=missing_df)
    plt.title("Missing Values Percentage by Column")
    plt.xlabel("Percentage (%)")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.show()


def numeric_columns_for_eda(df, exclude_cols=None):
    """Return numeric columns used in the univariate EDA plots."""
    exclude_cols = set(exclude_cols or [])
    return [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in exclude_cols
    ]


def plot_numeric_distributions(df, columns, color="#1B4F72"):
    """Plot histograms for numerical variables."""
    if not columns:
        print("No numerical columns available for plotting.")
        return

    fig, axes = plt.subplots(
        nrows=(len(columns) + 1) // 2,
        ncols=2,
        figsize=(16, len(columns) * 2),
    )
    axes = np.ravel(axes)

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color=color, alpha=0.8)
        axes[i].set_title(f"{col} distribution", fontsize=13, fontweight="bold", pad=10)
        axes[i].set_xlabel("")
        axes[i].grid(axis="y", linestyle="--", alpha=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def get_skewness_table(df, columns):
    """Return skewness sorted by absolute magnitude."""
    skewness_df = pd.DataFrame({
        "feature": columns,
        "skewness": [df[col].skew() for col in columns],
    })
    return skewness_df.sort_values("skewness", key=abs, ascending=False)


def plot_numeric_boxplots(df, columns):
    """Plot boxplots for numerical variables."""
    if not columns:
        print("No numerical columns available for plotting.")
        return

    fig, axes = plt.subplots(
        nrows=(len(columns) + 1) // 2,
        ncols=2,
        figsize=(16, len(columns) * 2),
    )
    axes = np.ravel(axes)

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"{col} boxplot")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def set_future_years_to_missing(df, current_year, year_col="year_first_transaction"):
    """Set future transaction years to missing and keep all rows."""
    out = df.copy()
    future_mask = out[year_col] > current_year
    out.loc[future_mask, year_col] = np.nan
    return out, int(future_mask.sum())


def set_invalid_promotion_to_missing(df, col="percentage_of_products_bought_promotion"):
    """Set promotion percentages outside [0, 1] to missing."""
    out = df.copy()
    invalid_mask = (out[col] < 0.0) | (out[col] > 1.0)
    out.loc[invalid_mask, col] = np.nan
    return out, int(invalid_mask.sum())


def parse_birthdate(df, col="customer_birthdate"):
    """Parse customer birthdate values using the source dataset format."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], format="%m/%d/%Y %I:%M %p", errors="coerce")
    return out


def coerce_numeric_columns(df, columns):
    """Convert selected columns to numeric values."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_cyclic_hour(df):
    """Plot the sine/cosine representation of the typical purchase hour."""
    plt.figure(figsize=(6, 6))
    plt.scatter(df["typical_hour_sin"], df["typical_hour_cos"])
    plt.title("Typical Hour Representation")
    plt.xlabel("sin")
    plt.ylabel("cos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fill_zero_count_columns(df, columns):
    """Fill missing count columns where absence is interpreted as zero."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].fillna(0)
    return out


def encode_loyalty_flag(df, source_col="loyalty_card_number", target_col="customer_loyalty_flag"):
    """Convert loyalty-card numbers into a binary loyalty flag."""
    out = df.copy()
    out.rename(columns={source_col: target_col}, inplace=True)
    out[target_col] = out[target_col].fillna(0).apply(lambda x: 1 if x != 0 else 0)
    return out


def add_customer_age(df, current_date=None, birthdate_col="customer_birthdate"):
    """Create customer_age and set implausible ages to missing."""
    out = df.copy()
    current_date = current_date or pd.Timestamp.now().normalize()
    out["customer_age"] = (current_date - out[birthdate_col]).dt.days // 365

    invalid_age = (out["customer_age"] < 16) | (out["customer_age"] > 100)
    summary = {
        "min_age": out["customer_age"].min(),
        "max_age": out["customer_age"].max(),
        "invalid_count": int(invalid_age.sum()),
    }

    out.loc[invalid_age, "customer_age"] = np.nan
    out.drop(columns=[birthdate_col], inplace=True)
    return out, summary


def plot_age_distribution(df):
    """Plot the customer age distribution."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df["customer_age"], kde=True)
    plt.title("Customer Age Distribution")
    plt.tight_layout()
    plt.show()


def encode_gender(df, source_col="customer_gender", target_col="is_male"):
    """Convert gender into a binary indicator and keep the original column for review."""
    out = df.copy()
    out[target_col] = out[source_col].map({"male": 1, "female": 0})
    return out


def cast_nullable_int(df, columns):
    """Round selected columns and cast them to pandas nullable integers."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].round().astype("Int64")
    return out


def make_absolute_spend_view(df):
    """Return the renamed absolute-spend feature view used for inspection."""
    out = df.copy()
    rename_map = {
        "customer_age": "age",
        "typical_hour_sin": "hour_sin",
        "typical_hour_cos": "hour_cos",
        "education_level": "years_education",
        "tenure": "years_since_first_transaction",
        "customer_loyalty_flag": "has_loyalty_card",
        "is_male": "customer_gender",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    spend_cols = [c for c in out.columns if c.startswith("lifetime_spend_")]
    selected_cols = [
        "kids_home",
        "teens_home",
        "number_complaints",
        "distinct_stores_visited",
        *spend_cols,
        "lifetime_total_distinct_products",
        "percentage_of_products_bought_promotion",
        "latitude",
        "longitude",
        "age",
        "hour_sin",
        "hour_cos",
        "years_education",
        "years_since_first_transaction",
        "has_loyalty_card",
        "customer_gender",
    ]
    selected_cols = [c for c in selected_cols if c in out.columns]
    return out[selected_cols]


def plot_outlier_diagnostics(df, columns, color="#1B4F72"):
    """Plot boxplot and histogram diagnostics for selected columns."""
    columns = [col for col in columns if col in df.columns]
    if not columns:
        print("No columns available for outlier visualization.")
        return

    fig, axes = plt.subplots(nrows=len(columns), ncols=2, figsize=(16, len(columns) * 4))
    if len(columns) == 1:
        axes = np.array([axes])

    for i, col in enumerate(columns):
        sns.boxplot(
            x=df[col],
            ax=axes[i, 0],
            color=color,
            flierprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "red",
                "markersize": 5,
                "alpha": 0.5,
            },
        )
        axes[i, 0].set_title(f"{col} - Outlier Detection (Boxplot)", fontsize=12, fontweight="bold")
        axes[i, 0].set_xlabel("")

        sns.histplot(df[col], kde=True, ax=axes[i, 1], color=color)
        axes[i, 1].set_title(f"{col} - Distribution", fontsize=12, fontweight="bold")
        axes[i, 1].set_xlabel("")

    plt.tight_layout()
    plt.show()


def correlation_columns(df, exclude_cols=None):
    """Return numeric columns to use in the correlation heatmap."""
    exclude_cols = set(exclude_cols or [])
    return [
        col for col in df.select_dtypes(include=np.number).columns.tolist()
        if col not in exclude_cols
    ]


def separate_outliers_and_impute_regular(
    df,
    index_col="customer_id",
    iqr_multiplier=3.0,
    min_flagged_features=2,
    max_remove_pct=5.0,
):
    """Separate raw outliers and impute the regular customer base."""
    out = df.copy()
    plot_missing_heatmap(out, "Missing Values Before Imputation")
    print(get_missing_report(out).to_string())

    if index_col in out.columns:
        out = out.set_index(index_col, drop=True)

    raw_spend_cols = [c for c in out.columns if c.startswith("lifetime_spend_")]
    tailed_cols = [
        c for c in ["customer_age", "distinct_stores_visited", "lifetime_total_distinct_products"]
        if c in out.columns
    ]

    regular_df, outlier_df, outlier_summary = split_outliers_iqr(
        out,
        feature_cols=raw_spend_cols + tailed_cols,
        iqr_multiplier=iqr_multiplier,
        min_flagged_features=min_flagged_features,
        max_remove_pct=max_remove_pct,
    )

    before_imputation = regular_df.copy()
    exclude_cols = [
        c for c in ["customer_loyalty_flag", "is_male", "education_level"]
        if c in regular_df.columns
    ]

    regular_df = apply_knn_imputation(
        regular_df,
        n_neighbors=5,
        exclude_cols=exclude_cols,
    )

    imputed_columns = [
        c for c in regular_df.select_dtypes(include="number").columns
        if c not in exclude_cols
    ]

    validate_imputation(before_imputation, regular_df, imputed_columns)
    print("Remaining missing values:", regular_df.isnull().sum().sum())
    plot_missing_heatmap(regular_df, "Missing Values After Imputation")

    return regular_df, outlier_df, outlier_summary

def align_outlier_features(outlier_df, reference_df, current_year):
    """Apply the final imputation and feature-engineering steps to the outlier set."""
    if outlier_df is None or len(outlier_df) == 0:
        return outlier_df

    out_exclude = [
        c for c in ["customer_loyalty_flag", "is_male", "education_level"]
        if c in outlier_df.columns
    ]
    out = apply_knn_imputation(outlier_df, n_neighbors=5, exclude_cols=out_exclude)
    out = engineer_clustering_features(out, current_year, keep_absolute_spend=True)

    same_cols = set(out.columns) == set(reference_df.columns)
    print("Outlier columns match regular columns:", same_cols)
    print("Outlier dataset shape:", out.shape)
    return out


def export_preprocessing_outputs(regular_df, outlier_df=None, output_dir="../datasets"):
    """Export the unscaled regular and outlier datasets used by clustering."""
    os.makedirs(output_dir, exist_ok=True)

    regular_path = os.path.join(output_dir, "info_clustering_unscaled.csv")
    regular_df.to_csv(regular_path, index=True)
    print("Regular dataset exported UNSCALED:", regular_df.shape)

    if outlier_df is not None:
        outlier_path = os.path.join(output_dir, "outlier_dataset.csv")
        outlier_df.to_csv(outlier_path, index=True)
        print("Outlier dataset exported UNSCALED:", outlier_df.shape)

    spend_cols = [c for c in regular_df.columns if c.startswith("lifetime_spend_")]
    print("Absolute lifetime spend columns exported:", len(spend_cols))
    print(spend_cols)
    print("All exported columns:")
    print(list(regular_df.columns))

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
