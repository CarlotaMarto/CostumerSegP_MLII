#imports 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import DBSCAN


def read_customer_data(filepath="customer_info.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

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

def customer_data_shape(customer_info):
    data_shape = customer_info.shape
    return data_shape

def customer_data_null(customer_info):
    null_values = customer_info.isnull().sum()
    return null_values


def customer_data_duplicates(customer_info):
    duplicates = customer_info.duplicated().sum()
    return duplicates

def customer_data_describe(customer_info):
    data_describe = customer_info.describe()
    return data_describe

def customer_data_columns(customer_info):
    columns = customer_info.columns
    return columns

def customer_data_unique(customer_info):
    unique = customer_info.nunique()
    return unique

def customer_data_value_counts(customer_info):
    value_counts = customer_info.value_counts()
    return value_counts


def add_age_column(customer_info: pd.DataFrame,
                   reference_date: str = '2025-06-09') -> pd.DataFrame:
    """
    Adds an integer `age` column (years) to **the same DataFrame** and
    returns it. No internal copy is made, so the original object passed
    in will now contain the new column.

    * Birth‑dates are parsed with the explicit format
      ``%m/%d/%Y %I:%M %p`` (e.g. ``07/15/1983 12:00 AM``).
    * Missing ages are imputed with the global median.
    * The intermediate birth‑date column is dropped for tidiness.
    """

    customer_info['customer_birthdate'] = pd.to_datetime(
        customer_info['customer_birthdate'],
        format='%m/%d/%Y %I:%M %p',
        errors='coerce'
    )

    ref_dt = pd.Timestamp(reference_date)
    customer_info['age'] = (
        (ref_dt - customer_info['customer_birthdate']).dt.days / 365.25
    )

    return customer_info


def drop_constant_columns(df, skip_cols, verbose):
    """
    Remove columns that carry **no variation** (all values identical or all NA).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to prune *in place*.
    skip_cols : list[str] | None, optional
        Columns that must be preserved even if constant.
    verbose : bool, default True
        If *True*, prints the list of columns removed.

    Returns
    -------
    pd.DataFrame
        The same DataFrame object, now without constant columns, enabling
        chaining like ``df = drop_constant_columns(df)``.
    """
    if skip_cols is None:
        skip_cols = []

    nunique = df.nunique(dropna=True)
    constant_cols = [col for col, n in nunique.items()
                     if n <= 1 and col not in skip_cols]

    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        if verbose:
            print(f"Dropped {len(constant_cols)} constant column(s): {constant_cols}")
    elif verbose:
        print("No constant columns found.")

    return df

def remove_semi_constant_features(df, unique_threshold=5, skew_threshold=0.9):
    semi_constant_columns = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        if unique_count <= unique_threshold:
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > skew_threshold:
                semi_constant_columns.append(col)
    
    df = df.drop(columns=semi_constant_columns)
    
    return df, semi_constant_columns


def add_loyalty_flag(df, card_col='loyalty_card_number', flag_name='has_loyalty_card'):
    """Add 1 if the customer has a loyalty card, else 0; drop the raw ID."""
    df[flag_name] = df[card_col].notna().astype('int8')
    return df 


def add_years_of_education(df: pd.DataFrame, name_col: str = 'customer_name') -> pd.DataFrame:
    """
    Adds a `years_education` column inferred from degree titles in the customer name.

    Heuristic mapping:
    - 'doctor'   ➜ 22 years
    - 'master'   ➜ 17 years
    - 'bachelor' ➜ 15 years
    - 'unknown'  ➜ 12 years
    """
    token_map = {
        'phd': 'doctor', 'dr.': 'doctor',
        'msc': 'master', 'm.sc': 'master', 'ms': 'master', 'ma': 'master',
        'bsc': 'bachelor', 'bs': 'bachelor', 'ba': 'bachelor', 'lic.': 'bachelor'
    }
    level_to_years = {
        'doctor': 22,
        'master': 17,
        'bachelor': 15,
        'unknown': 12
    }

    def detect_level(text: str):
        if not isinstance(text, str):
            return 'unknown'
        t = text.lower()
        for token, level in token_map.items():
            pattern = rf'(?:^|\s){re.escape(token)}(?:\s|$|[,.])'
            if re.search(pattern, t):
                return level
        return 'unknown'

    levels = df[name_col].apply(detect_level)
    df['years_education'] = levels.map(level_to_years).astype('int16')
    return df


def add_years_since_first_transaction(
    df: pd.DataFrame,
    year_col: str = "year_first_transaction",
    new_col: str = "years_since_first_transaction"
) -> pd.DataFrame:
    """Add a column for years since the customer's first transaction."""
    current_year = datetime.today().year
    df[new_col] = current_year - df[year_col]
    return df



def apply_threshold_and_outlier_filter(
    df: pd.DataFrame,
    bounded_caps: dict | None = None,
    excluded_cols: list | None = None,
    iqr_k: float = 1.75,
    som_percentile: float = 90.0,
    dbscan_eps: float = 0.75,
    min_remove_pct: float = 0.0,
    max_remove_pct: float = 5.0,
    id_col: str = "customer_id",            
):
    """
    ( … docstring unchanged … )
    Returns
    -------
    df_cleaned : DataFrame      # inliers
    df_outliers : DataFrame      # rows flagged by *all* rules
    """
    df = df.copy()

    if bounded_caps is None:
        bounded_caps = {"kids_home": 3, "teens_home": 2,
                        "number_complaints": 2, "distinct_stores_visited": 6}

    for col, cap in bounded_caps.items():
        if col in df.columns:
            n_clip = (df[col] > cap).sum()
            df[col] = np.minimum(df[col], cap)
            print(f"[bounding] {col:<25s} capped @ {cap:<2} | clipped: {n_clip:5d}")

    if excluded_cols is None:
        excluded_cols = [
            "has_loyalty_card", "hour_sin", "hour_cos",
            "latitude", "longitude", "year_first_transaction",
        ]
    excluded_cols = [c for c in excluded_cols if c != id_col]

    numeric_cols = [c for c in df.select_dtypes("number") if c not in excluded_cols]
    print(f"[info] numeric cols used: {len(numeric_cols)}")

    iqr_mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        iqr_mask |= (df[col] < lo) | (df[col] > hi)
    print(f"[IQR] flagged: {iqr_mask.sum()} rows")

    rows_ok = df[numeric_cols].dropna().index
    X_std   = StandardScaler().fit_transform(df.loc[rows_ok, numeric_cols])

    db = DBSCAN(eps=dbscan_eps, min_samples=len(numeric_cols) + 1)
    db_mask = pd.Series(False, index=df.index)
    db_mask.loc[rows_ok] = db.fit_predict(X_std) == -1
    print(f"[DBSCAN] flagged: {db_mask.sum()}")

    from minisom import MiniSom
    som = MiniSom(10, 10, X_std.shape[1], sigma=1., learning_rate=.5, random_seed=0)
    som.train(X_std, 500, verbose=False)
    qe  = np.array([som.quantization_error([x]) for x in X_std])
    thr = np.percentile(qe, som_percentile)
    som_mask = pd.Series(False, index=df.index)
    som_mask.loc[rows_ok] = qe > thr
    print(f"[SOM] flagged: {som_mask.sum()}")

    final_mask = iqr_mask & db_mask & som_mask
    pct_drop   = final_mask.mean() * 100
    print(f"[ALL] to drop: {final_mask.sum()} rows ({pct_drop:.2f}%)")

    if pct_drop > max_remove_pct:
        raise ValueError(f"Would drop {pct_drop:.2f}% (> {max_remove_pct}%) – abort.")
    if pct_drop < min_remove_pct:
        print(f"[warn] only {pct_drop:.2f}% dropped (< {min_remove_pct}%)")

    df_cleaned  = df.loc[~final_mask].copy()
    df_outliers = df.loc[ final_mask].copy()

    if id_col in df.columns and id_col not in df_outliers.columns:
        df_outliers[id_col] = df[id_col]

    return df_cleaned, df_outliers


    
def cor_heatmap(corr, figsize=(14, 12), cmap="vlag"):
    mask = np.triu(np.ones_like(corr, dtype=bool)) 
    plt.figure(figsize=figsize)
    sns.heatmap(
        data=corr,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.8,
        cbar_kws={'shrink': 0.75},
        vmin=-1,
        vmax=1
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Spearman Correlation Heatmap', fontsize=14, pad=12)
    plt.tight_layout()
    plt.show()



def save_to_csv(df, path='final_preproc_dataset.csv', index=False):
    """
    Saves the DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : str, optional
        The file path where to save the CSV.
    index : bool, optional
        Whether to write row names (index). Default is False.
    """
    df.to_csv(path, index=index)
    print(f"File saved to: {path}")



def scale_standard(df: pd.DataFrame, binary_cols: list = None) -> pd.DataFrame:
    """Apply StandardScaler to numeric columns, excluding binary ones."""
    binary_cols = binary_cols or []
    scale_cols = [col for col in df.select_dtypes(include='number') if col not in binary_cols]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled, columns=scale_cols, index=df.index)

    return pd.concat([scaled_df, df[binary_cols]], axis=1)


def scale_robust(df: pd.DataFrame, binary_cols: list = None) -> pd.DataFrame:
    """Apply RobustScaler to numeric columns, excluding binary ones."""
    binary_cols = binary_cols or []
    scale_cols = [col for col in df.select_dtypes(include='number') if col not in binary_cols]

    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled, columns=scale_cols, index=df.index)

    return pd.concat([scaled_df, df[binary_cols]], axis=1)


def scale_minmax(df: pd.DataFrame, binary_cols: list = None) -> pd.DataFrame:
    """Apply MinMaxScaler to numeric columns, excluding binary ones."""
    binary_cols = binary_cols or []
    scale_cols = [col for col in df.select_dtypes(include='number') if col not in binary_cols]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled, columns=scale_cols, index=df.index)

    return pd.concat([scaled_df, df[binary_cols]], axis=1)


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with the median of each column."""
    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)

def impute_knn (df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using KNN imputation."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns, index=df.index)
