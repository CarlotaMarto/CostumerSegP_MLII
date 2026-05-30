"""
Clustering.py  —  Helper functions for the Customer Segmentation project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.cluster.hierarchy import dendrogram, fcluster


# =============================================================================
# Data loading
# =============================================================================

def read_newdata(filepath="../datasets/info_clustering_unscaled.csv"):
    """Load the unscaled clustering dataset exported by the EDA notebook."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]:,} customers | {df.shape[1]} features")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def read_outliers(filepath="../datasets/outlier_dataset.csv"):
    """Load the outlier dataset to be assigned to clusters after modelling."""
    try:
        df = pd.read_csv(filepath)
        print(f"Outliers loaded: {df.shape[0]:,} customers | {df.shape[1]} features")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


# =============================================================================
# Internal scaling helper
# =============================================================================

def _scale(df, exclude_cols=None, binary_cols=None, ordinal_cols=None,
           scaler_name="minmax"):
    """Build the scaled feature matrix.

    Strategy
    --------
    Continuous  → scaler_name  (minmax / robust / standard)
    Ordinal     → StandardScaler always.
      education_level is encoded as years (12/15/17/22). MinMaxScaler would
      stretch these values to dominate inter-customer distances. StandardScaler
      keeps the signal without that distortion.
    Binary (0/1) → kept as-is
    Excluded     → dropped

    Returns X (np.ndarray) and col_names (list).
    """
    exclude_cols = [c for c in (exclude_cols or []) if c in df.columns]
    binary_cols  = [c for c in (binary_cols  or []) if c in df.columns]
    ordinal_cols = [c for c in (ordinal_cols or []) if c in df.columns]

    _scalers = {"minmax": MinMaxScaler(), "robust": RobustScaler(),
                "standard": StandardScaler()}
    if scaler_name not in _scalers:
        raise ValueError(f"scaler_name must be one of {list(_scalers)}")

    X_df      = df.drop(columns=exclude_cols)
    cont_cols = [c for c in X_df.columns if c not in binary_cols + ordinal_cols]

    parts, names = [], []
    if cont_cols:
        parts.append(_scalers[scaler_name].fit_transform(X_df[cont_cols].astype(float)))
        names.extend(cont_cols)
    if ordinal_cols:
        parts.append(StandardScaler().fit_transform(X_df[ordinal_cols].astype(float)))
        names.extend(ordinal_cols)
    if binary_cols:
        parts.append(X_df[binary_cols].values.astype(float))
        names.extend(binary_cols)

    X = np.concatenate(parts, axis=1) if parts else X_df.values.astype(float)
    return X, names


# =============================================================================
# Scaler comparison
# =============================================================================

def compare_scalers(df, exclude_cols=None, binary_cols=None, ordinal_cols=None,
                    k_values=(4, 5, 6), random_state=42, include_hierarchical=False):
    """Compare Standard, MinMax and Robust scalers for K-Means (and optionally Hierarchical).

    Silhouette ↑  (higher is better).
    No PCA — clustering is in the original scaled feature space.
    """
    from scipy.cluster.hierarchy import linkage as _linkage, fcluster as _fcluster

    results = []
    algos   = ["K-Means"] + (["Hierarchical (Ward)"] if include_hierarchical else [])

    for sc_name in ("Standard", "MinMax", "Robust"):
        X_sc, _ = _scale(df, exclude_cols, binary_cols, ordinal_cols, sc_name.lower())
        for algo in algos:
            for k in k_values:
                if algo == "K-Means":
                    lbl = KMeans(n_clusters=k, random_state=random_state,
                                 n_init=20).fit_predict(X_sc)
                else:
                    km20  = KMeans(n_clusters=20, random_state=random_state,
                                   n_init=10).fit(X_sc)
                    Z     = _linkage(km20.cluster_centers_, method="ward")
                    macro = _fcluster(Z, t=k, criterion="maxclust") - 1
                    lbl   = macro[km20.labels_]

                sil = silhouette_score(X_sc, lbl,
                                       sample_size=min(5000, len(lbl)),
                                       random_state=random_state)
                results.append({"Algorithm": algo, "Scaler": sc_name,
                                 "k": k, "Silhouette": round(sil, 4)})

    return (pd.DataFrame(results)
            .sort_values("Silhouette", ascending=False)
            .reset_index(drop=True))


def plot_scaler_comparison(comparison_df):
    """Line chart of Silhouette by scaler, algorithm and k.

    Solid lines = K-Means; dashed lines = Hierarchical Ward.
    """
    colors    = {"Standard": "#0047AB", "MinMax": "#ff7f0e", "Robust": "#2ca02c"}
    linestyle = {"K-Means": "-", "Hierarchical (Ward)": "--"}
    ks        = sorted(comparison_df["k"].unique())
    algos     = comparison_df["Algorithm"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in algos:
        for sc in comparison_df["Scaler"].unique():
            sub = comparison_df[(comparison_df["Scaler"] == sc) &
                                (comparison_df["Algorithm"] == algo)].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["k"], sub["Silhouette"], marker="o",
                    color=colors[sc], linestyle=linestyle.get(algo, "-"),
                    label=f"{sc} ({algo})" if len(algos) > 1 else sc)

    ax.set_title("Scaler Comparison — Silhouette Score (original feature space, no PCA)")
    ax.set_xlabel("k"); ax.set_xticks(ks)
    ax.set_ylabel("Silhouette ↑")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# K-Means
# =============================================================================

def run_kmeans(df, k, exclude_cols=None, binary_cols=None, ordinal_cols=None,
               scaler_name="minmax", random_state=42):
    """Run K-Means with scaling on df, excluding specified columns.

    Returns
    -------
    labels    : np.ndarray  — cluster assignment per customer
    centroids : np.ndarray  — cluster centres in original (unscaled) units
    score     : float       — silhouette score in the scaled space
    X_scaled  : np.ndarray  — scaled matrix (pass to silhouette_plot for consistency)
    """
    X_scaled, _ = _scale(df, exclude_cols, binary_cols, ordinal_cols, scaler_name)
    model        = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels       = model.fit_predict(X_scaled)
    score        = silhouette_score(X_scaled, labels,
                                     sample_size=min(5000, len(labels)),
                                     random_state=random_state)

    # Centroids in original (unscaled) units for readable business profiles
    _excl     = [c for c in (exclude_cols or []) if c in df.columns]
    keep_cols = [c for c in df.columns if c not in _excl]
    centroids = df[keep_cols].groupby(labels).mean().values

    return labels, centroids, score, X_scaled


def elbow_and_silhouette(df, max_k=12, exclude_cols=None, binary_cols=None,
                          ordinal_cols=None, scaler_name="minmax", random_state=42):
    """Plot Elbow and Silhouette curves for k = 2 to max_k with the chosen scaler.

    Both curves use the same scaled feature space as run_kmeans() so the
    silhouette values here match those returned by run_kmeans() exactly.
    """
    X, _ = _scale(df, exclude_cols, binary_cols, ordinal_cols, scaler_name)
    inertias, silhouettes = [], []

    for k in range(2, max_k + 1):
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        score = silhouette_score( X, labels, sample_size=min(5000, len(labels)), random_state=random_state)
        silhouettes.append(score)
        print(f"K={k}: Inertia={km.inertia_:.0f}, Silhouette={score:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.plot(range(2, max_k + 1), inertias, marker="o", color="tab:blue", label="Inertia")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:green")
    ax2.plot(range(2, max_k + 1), silhouettes, marker="x", linestyle="--",
             color="tab:green", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title(f"Elbow and Silhouette Scores — {scaler_name.capitalize()} Scaler")
    fig.tight_layout()
    plt.show()
    return inertias, silhouettes


# =============================================================================
# DBSCAN
# =============================================================================

def explore_dbscan(X, eps_values=None, min_samples=10):
    """Try multiple eps values and return a summary table sorted by silhouette.

    Only configurations with ≥ 2 clusters and < 50% noise are scored.
    """
    if eps_values is None:
        eps_values = np.arange(0.5, 5.0, 0.5)

    data    = X.values if hasattr(X, "values") else np.array(X)
    results = []

    for eps in eps_values:
        labels     = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct  = round((labels == -1).sum() / len(labels) * 100, 1)
        score      = np.nan
        valid      = labels != -1
        if n_clusters >= 2 and noise_pct < 50 and valid.sum() > n_clusters:
            score = round(silhouette_score(data[valid], labels[valid]), 4)
        results.append({"eps": eps, "n_clusters": n_clusters,
                        "noise_pct": noise_pct, "silhouette": score})

    return (pd.DataFrame(results)
            .sort_values("silhouette", ascending=False, na_position="last")
            .reset_index(drop=True))


def run_dbscan(X, eps, min_samples=10):
    """Run DBSCAN with the chosen parameters.

    Returns labels, silhouette score (NaN if < 2 valid clusters) and noise %.
    """
    data      = X.values if hasattr(X, "values") else np.array(X)
    labels    = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
    noise_pct = round((labels == -1).sum() / len(labels) * 100, 1)
    valid     = labels != -1
    n_valid   = len(set(labels[valid]))
    score     = (silhouette_score(data[valid], labels[valid])
                 if n_valid >= 2 and valid.sum() > n_valid else np.nan)
    return labels, score, noise_pct


# =============================================================================
# Mean Shift
# =============================================================================

def explore_meanshift(X, quantile_range=None, sample_size=5000, random_state=42):
    """Try multiple bandwidth quantiles and return a summary table sorted by silhouette.

    Runs on a sample for speed. Lower quantile → smaller bandwidth → more clusters.
    """
    if quantile_range is None:
        quantile_range = np.arange(0.05, 0.35, 0.05)

    data = X.values if hasattr(X, "values") else np.array(X)
    np.random.seed(random_state)
    idx      = np.random.choice(len(data), size=min(sample_size, len(data)), replace=False)
    X_sample = data[idx]
    results  = []

    for q in quantile_range:
        bw = estimate_bandwidth(X_sample, quantile=q,
                                n_samples=min(1000, len(X_sample)),
                                random_state=random_state)
        if bw <= 0:
            continue
        labels     = MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X_sample)
        n_clusters = len(set(labels))
        score      = np.nan
        if n_clusters >= 2:
            score = round(silhouette_score(X_sample, labels), 4)
        results.append({"quantile": round(q, 3), "bandwidth": round(bw, 4),
                        "n_clusters": n_clusters, "silhouette": score})

    return (pd.DataFrame(results)
            .sort_values("silhouette", ascending=False, na_position="last")
            .reset_index(drop=True))


def run_meanshift(X, quantile, sample_size=5000, random_state=42):
    """Run Mean Shift with the chosen bandwidth quantile on a sample.

    Returns labels, silhouette score (NaN if < 2 clusters) and cluster count.
    """
    data = X.values if hasattr(X, "values") else np.array(X)
    np.random.seed(random_state)
    idx      = np.random.choice(len(data), size=min(sample_size, len(data)), replace=False)
    X_sample = data[idx]

    bw = estimate_bandwidth(X_sample, quantile=quantile,
                            n_samples=min(1000, len(X_sample)),
                            random_state=random_state)
    print(f"  Estimated bandwidth (quantile={quantile}): {bw:.4f}")
    labels     = MeanShift(bandwidth=bw, bin_seeding=True).fit_predict(X_sample)
    n_clusters = len(set(labels))
    score      = silhouette_score(X_sample, labels) if n_clusters >= 2 else np.nan
    return labels, score, n_clusters


# =============================================================================
# Model comparison
# =============================================================================

def compare_models(scores_dict):
    """Build a sorted comparison table from {model_name: silhouette_score}."""
    rows = [{"Model": name,
             "Silhouette": round(score, 4) if not np.isnan(score) else np.nan}
            for name, score in scores_dict.items()]
    return (pd.DataFrame(rows)
            .sort_values("Silhouette", ascending=False, na_position="last")
            .reset_index(drop=True))


# =============================================================================
# Visualisations
# =============================================================================

def get_color_map(labels, cmap_name="tab10"):
    """Consistent colour map for cluster labels."""
    unique    = sorted(np.unique(labels))
    cmap      = cm.get_cmap(cmap_name)
    color_map = {l: cmap(i % 10) for i, l in enumerate(unique)}
    return [color_map[l] for l in labels]


def silhouette_plot(X, cluster_labels, title="Silhouette Plot", cmap_name="tab10"):
    """Per-sample silhouette plot grouped by cluster.

    Pass the same X_scaled returned by run_kmeans() so the average matches exactly.
    Wide positive bars = well-defined cluster. Red dashed line = global average.
    """
    avg     = silhouette_score(X, cluster_labels)
    vals    = silhouette_samples(X, cluster_labels)
    unique  = sorted(np.unique(cluster_labels))
    cmap    = cm.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10

    for i, label in enumerate(unique):
        v       = np.sort(vals[cluster_labels == label])
        y_upper = y_lower + len(v)
        color   = cmap(i % 10)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, v,
                         facecolor=color, edgecolor=color, alpha=0.7,
                         label=f"Cluster {label}")
        ax.text(-0.05, y_lower + 0.5 * len(v), str(label))
        y_lower = y_upper + 10

    ax.axvline(x=avg, color="red", linestyle="--", label=f"Avg = {avg:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_pca(embedding, labels=None, title="PCA Projection",
                  xlabel="PC 1", ylabel="PC 2", cmap_name="tab10"):
    """2D PCA scatter coloured by cluster — visualisation only."""
    plt.figure(figsize=(10, 8))
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    else:
        unique = sorted(np.unique(labels))
        colors = get_color_map(labels, cmap_name)
        for label in unique:
            idx   = labels == label
            color = colors[np.where(labels == label)[0][0]]
            plt.scatter(embedding[idx, 0], embedding[idx, 1],
                        c=[color], s=10, label=f"Cluster {label}")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True); plt.tight_layout(); plt.show()


def visualize_umap(embedding, labels=None, title="UMAP Projection",
                   xlabel="Component 1", ylabel="Component 2", cmap_name="tab10"):
    """2D UMAP scatter coloured by cluster — visualisation only."""
    plt.figure(figsize=(10, 8))
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    else:
        unique = sorted(np.unique(labels))
        colors = get_color_map(labels, cmap_name)
        for label in unique:
            idx   = labels == label
            color = colors[np.where(labels == label)[0][0]]
            plt.scatter(embedding[idx, 0], embedding[idx, 1],
                        c=[color], s=10, label=f"Cluster {label}")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True); plt.tight_layout(); plt.show()


# =============================================================================
# Dendrogram helpers (kept for compatibility)
# =============================================================================

def plot_dendrogram(linked, title="Hierarchical Clustering on K-Means Centroids"):
    """Plot a dendrogram from a scipy linkage matrix."""
    plt.figure(figsize=(10, 6))
    dendrogram(linked, orientation="top",
               distance_sort="descending", show_leaf_counts=True)
    plt.title(title); plt.ylabel("Ward distance")
    plt.tight_layout(); plt.show()


def assign_macro_clusters(linked, num_clusters):
    """Cut the dendrogram and return 1-indexed macro-cluster labels."""
    return fcluster(linked, t=num_clusters, criterion="maxclust")
