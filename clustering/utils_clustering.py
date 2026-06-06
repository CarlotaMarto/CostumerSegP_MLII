"""Utilities for the customer segmentation clustering workflow.

This module centralises feature construction, scaling, clustering diagnostics,
robustness checks and segment profiling used in the clustering notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix
from scipy.cluster.hierarchy import dendrogram

PROJECT_PALETTE = [
    "#B87540",
    "#B2543D",
    "#7E6A43",
    "#A8B7BA",
    "#D8C0B4",
    "#C8AB8C",
    "#5A3516",
    "#B98F70",
]
CLUSTER_PALETTE = [
    "#B87540",
    "#B2543D",
    "#7E6A43",
    "#78969B",
    "#D08F78",
    "#C09A72",
    "#5A3516",
    "#2F2116",
]
MAIN_COLOR = "#B87540"
ACCENT_COLOR = "#B2543D"
NOTE_COLOR = "#7E6A43"
SECONDARY_COLOR = "#A8B7BA"
LIGHT_COLOR = "#F3EEE6"
DARK_COLOR = "#5A3516"


def sequential_cmap():
    return LinearSegmentedColormap.from_list(
        "project_sequential",
        [LIGHT_COLOR, "#C8AB8C", MAIN_COLOR, ACCENT_COLOR, DARK_COLOR],
    )


def diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "project_diverging",
        [ACCENT_COLOR, LIGHT_COLOR, SECONDARY_COLOR],
    )


def cluster_cmap():
    return ListedColormap(CLUSTER_PALETTE)


# ============================================================
# Feature selection for the clustering distance
# ============================================================

def get_profiling_features(df, distance_cols):
    """Return every numeric column NOT used in the distance (for profiling)."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in distance_cols]


# ============================================================
# Scaling (fit once on regular customers, reuse for outliers)
# ============================================================

def transform_with_scaler(df, feature_cols, scaler):
    """Project new rows (e.g. outliers) using an already-fitted scaler."""
    return scaler.transform(df[feature_cols].astype(float))


# ============================================================
# Choosing the number of clusters
# ============================================================

def kmeans_elbow(X, k_range=range(1, 11), random_state=0, n_init=30):
    """Compute KMeans inertia across k (the elbow / dispersion curve)."""
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
        inertia.append(km.inertia_)
    return list(k_range), inertia


def plot_elbow(k_values, inertia, cutoffs=None):
    """Plot the inertia elbow, optionally marking candidate cut points."""
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, inertia, marker="o", color=MAIN_COLOR)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Dispersion (inertia)")
    plt.title("Elbow Method - K-Means")
    plt.xticks(list(k_values))
    if cutoffs:
        for c in cutoffs:
            plt.axvline(c, color=ACCENT_COLOR, linestyle="--", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# Fitting the final solutions
# ============================================================

def fit_kmeans(X, k, random_state=0, n_init=30):
    """Fit and return a KMeans model (use .labels_ / .predict / .cluster_centers_)."""
    return KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)


def fit_hierarchical_sample(X, k, sample_size=5000, linkage="ward",
                            random_state=0):
    """Fit agglomerative clustering on a random sample."""
    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X[idx])
    return idx, labels


def fit_dendrogram_model(X, sample_size=5000, linkage="ward", random_state=0):
    """Fit a full-tree agglomerative model on a sample for dendrogram plots."""
    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    model = AgglomerativeClustering(
        linkage=linkage, distance_threshold=0, n_clusters=None
    ).fit(X[idx])
    return idx, model


def plot_dendrogram(model, **kwargs):
    """Plot a dendrogram from a fitted full-tree AgglomerativeClustering model."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        counts[i] = sum(
            1 if child_idx < n_samples else counts[child_idx - n_samples]
            for child_idx in merge
        )

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def dbscan_grid(X, eps_values=None, min_samples_values=None, sample_size=8000,
                random_state=0):
    """Evaluate DBSCAN on a sample for several eps/min_samples settings.

    DBSCAN can be useful as a density-based benchmark, but it often labels many
    customers as noise in high-dimensional customer data. The silhouette is
    computed only on non-noise observations when at least two clusters exist.
    """
    eps_values = eps_values or [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    min_samples_values = min_samples_values or [5, 10, 20, 40]

    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    Xs = X[idx]

    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
            non_noise = labels != -1
            n_clusters = len(set(labels[non_noise]))
            noise_pct = float((~non_noise).mean() * 100)

            sil = np.nan
            if n_clusters >= 2 and non_noise.sum() > n_clusters:
                sil = float(silhouette_score(Xs[non_noise], labels[non_noise]))

            rows.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_pct": round(noise_pct, 2),
                "silhouette_non_noise": np.nan if np.isnan(sil) else round(sil, 4),
            })
    return pd.DataFrame(rows).sort_values(
        ["silhouette_non_noise", "noise_pct"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def fit_dbscan(X, eps, min_samples):
    """Fit DBSCAN on the full matrix and return labels."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def kmeans_k_benchmark(X, k_range=range(2, 11), random_state=0,
                       n_init=10, sample_size=8000):
    """Evaluate KMeans across several k values."""
    X = np.asarray(X, dtype=float)
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state,
                        n_init=n_init).fit_predict(X)
        sil = silhouette_score(
            X, labels, sample_size=min(sample_size, len(labels)),
            random_state=random_state,
        )
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "method": "KMeans",
            "k": int(k),
            "silhouette": round(float(sil), 4),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "largest_share_%": round(float(sizes.max() / len(labels) * 100), 1),
        })
    return pd.DataFrame(rows)


def hierarchical_k_benchmark(X, k_range=range(2, 11), linkages=None,
                             sample_size=5000, random_state=0):
    """Evaluate agglomerative clustering across k and linkage methods on a sample."""
    X = np.asarray(X, dtype=float)
    linkages = linkages or ["ward", "complete", "average", "single"]
    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    Xs = X[idx]

    rows = []
    for linkage in linkages:
        for k in k_range:
            labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(Xs)
            sil = silhouette_score(Xs, labels)
            sizes = pd.Series(labels).value_counts()
            rows.append({
                "method": f"Agglomerative-{linkage}",
                "linkage": linkage,
                "k": int(k),
                "sample_size": int(n),
                "silhouette": round(float(sil), 4),
                "r2": round(clustering_r2(Xs, labels), 4),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "largest_share_%": round(float(sizes.max() / len(labels) * 100), 1),
            })
    return pd.DataFrame(rows).sort_values(
        ["silhouette", "r2"], ascending=[False, False]
    ).reset_index(drop=True)


def centroid_ward_k_benchmark(X, macro_k_range=range(2, 11), micro_k=20,
                              random_state=0, n_init=20, sample_size=8000):
    """Evaluate Ward macro-clusters built from KMeans micro-cluster centroids."""
    from scipy.cluster.hierarchy import linkage, fcluster

    X = np.asarray(X, dtype=float)
    micro = KMeans(n_clusters=micro_k, random_state=random_state,
                   n_init=n_init).fit(X)
    Z = linkage(micro.cluster_centers_, method="ward")

    rows = []
    for k in macro_k_range:
        centroid_labels = fcluster(Z, t=int(k), criterion="maxclust") - 1
        mapping = dict(zip(range(micro_k), centroid_labels))
        labels = np.array([mapping[x] for x in micro.labels_])
        sil = silhouette_score(
            X, labels, sample_size=min(sample_size, len(labels)),
            random_state=random_state,
        )
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "method": f"KMeans({micro_k}) + Ward centroids",
            "k": int(k),
            "micro_k": int(micro_k),
            "silhouette": round(float(sil), 4),
            "r2": round(clustering_r2(X, labels), 4),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "largest_share_%": round(float(sizes.max() / len(labels) * 100), 1),
        })
    return pd.DataFrame(rows)


def plot_method_k_benchmark(results_df, title="Method benchmark by k"):
    """Line plot of silhouette by k for several clustering methods."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=results_df, x="k", y="silhouette", hue="method", marker="o", palette=PROJECT_PALETTE)
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def benchmark_summary_table(kmeans_df, hierarchical_df, centroid_df):
    """Compact benchmark table without method-specific empty columns."""
    common_cols = [
        "method",
        "k",
        "silhouette",
        "min_cluster_size",
        "max_cluster_size",
        "largest_share_%",
    ]
    parts = []
    for df in [kmeans_df, hierarchical_df, centroid_df]:
        cols = [c for c in common_cols if c in df.columns]
        best = df.sort_values("silhouette", ascending=False).head(3)[cols]
        parts.append(best)
    return pd.concat(parts, ignore_index=True).sort_values(
        "silhouette", ascending=False
    ).reset_index(drop=True)


def valid_dbscan_results(dbscan_df):
    """Return DBSCAN settings where a silhouette can be computed."""
    cols = ["eps", "min_samples", "n_clusters", "noise_pct", "silhouette_non_noise"]
    valid = dbscan_df.dropna(subset=["silhouette_non_noise"]).copy()
    return valid[cols].sort_values(
        ["silhouette_non_noise", "noise_pct"], ascending=[False, True]
    ).reset_index(drop=True)

def compare_solutions(labels_a, labels_b, name_a="KMeans", name_b="Ward"):
    """Confusion matrix between two clustering label vectors (same rows)."""
    cm = confusion_matrix(labels_a, labels_b)
    k = cm.shape[0]
    return pd.DataFrame(
        cm,
        index=[f"{name_a} {i}" for i in range(k)],
        columns=[f"{name_b} {j}" for j in range(cm.shape[1])],
    )


def clustering_r2(X, labels):
    """Compute the R2-like between-cluster variance ratio for a clustering solution.

    R2 = SSB / SST, where SST is total sum of squares and SSB is the between-
    cluster sum of squares. Higher values indicate more homogeneous clusters
    relative to the total variance.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    overall = X.mean(axis=0)
    sst = ((X - overall) ** 2).sum()
    if sst == 0:
        return np.nan

    ssb = 0.0
    for lab in np.unique(labels):
        Xg = X[labels == lab]
        if len(Xg) == 0:
            continue
        diff = Xg.mean(axis=0) - overall
        ssb += len(Xg) * np.sum(diff ** 2)
    return float(ssb / sst)


def hierarchical_r2_grid(X, k_range=range(2, 11), linkages=None,
                         sample_size=5000, random_state=0):
    """Compare hierarchical linkage methods using an R2-like metric on a sample."""
    linkages = linkages or ["ward", "complete", "average", "single"]
    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    Xs = X[idx]

    rows = []
    for linkage in linkages:
        for k in k_range:
            labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(Xs)
            rows.append({
                "linkage": linkage,
                "k": int(k),
                "sample_size": int(n),
                "r2": round(clustering_r2(Xs, labels), 4),
            })
    return pd.DataFrame(rows)


def plot_hierarchical_r2(r2_df, title="Hierarchical R2 comparison"):
    """Line plot of hierarchical R2 values by k and linkage method."""
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=r2_df, x="k", y="r2", hue="linkage", marker="o", palette=PROJECT_PALETTE)
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("R2 = between-cluster SS / total SS")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def complaint_summary(df, label_col="cluster", complaint_col="number_complaints",
                      threshold=2):
    """Summarise complaint behaviour by cluster."""
    if complaint_col not in df.columns:
        raise ValueError(f"{complaint_col} not found in dataframe.")
    grouped = df.groupby(label_col)[complaint_col]
    out = pd.DataFrame({
        "customers": df[label_col].value_counts().sort_index(),
        "avg_complaints": grouped.mean().round(2),
        f"pct_complaints_ge_{threshold}": grouped.apply(lambda s: (s >= threshold).mean() * 100).round(1),
        "max_complaints": grouped.max(),
    })
    out.index.name = label_col
    return out


# ============================================================
# Profiling and visualisation of the final segments
# ============================================================

def profile_clusters(df, label_col, feature_cols, as_percent=False):
    """Mean profile by cluster, with an OVERALL row as baseline."""
    prof = df.groupby(label_col)[feature_cols].mean()
    prof.loc["OVERALL"] = df[feature_cols].mean()
    if as_percent:
        prof = prof * 100
    return prof.round(2)


def cluster_sizes(df, label_col):
    """Return a tidy size / share table per cluster."""
    sizes = df[label_col].value_counts().sort_index()
    out = pd.DataFrame({"customers": sizes,
                        "share_%": (sizes / sizes.sum() * 100).round(1)})
    out.index.name = label_col
    return out


def plot_cluster_sizes(df, label_col):
    """Bar chart of cluster sizes."""
    sizes = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=sizes.index.astype(str), y=sizes.values, color=MAIN_COLOR)
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.title("Customers per cluster")
    plt.show()


def plot_profile_heatmap(profile_df, title="Cluster profile"):
    """Heatmap of a profile table (clusters x features). Drop OVERALL row first."""
    data = profile_df.drop(index="OVERALL", errors="ignore")
    plt.figure(figsize=(max(8, data.shape[1] * 0.9), max(4, data.shape[0] * 0.7)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap=sequential_cmap(), linewidths=0.5,
                cbar_kws={"shrink": 0.6})
    plt.title(title)
    plt.ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_cluster_mean_comparison(profile_df, title="Cluster mean comparison"):
    """Dot plot of cluster means after min-max scaling each feature."""
    data = profile_df.copy()
    if "OVERALL" not in data.index:
        data.loc["OVERALL"] = data.mean()

    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(data.astype(float)),
        index=data.index.astype(str),
        columns=data.columns,
    )
    long = scaled.reset_index(names="cluster").melt(
        id_vars="cluster", var_name="feature", value_name="scaled_mean"
    )

    plt.figure(figsize=(10, max(5, len(data.columns) * 0.38)))
    sns.scatterplot(
        data=long, x="scaled_mean", y="feature", hue="cluster",
        s=90, alpha=0.9,
    )
    plt.title(title)
    plt.xlabel("Scaled mean within each feature (0 = lowest, 1 = highest)")
    plt.ylabel("Feature")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.show()
    return scaled.round(2)


def plot_boxplot_grid(data, variables, label_col="cluster", max_cols=3,
                      color=SECONDARY_COLOR):
    """Grid of boxplots for selected variables by cluster."""
    variables = [v for v in variables if v in data.columns]
    if not variables:
        raise ValueError("No requested variables were found in the dataframe.")

    n_cols = min(max_cols, len(variables))
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, variables):
        sns.boxplot(data=data, x=label_col, y=col, ax=ax, color=color, fliersize=1.5)
        ax.set_title(col)
        ax.set_xlabel("Cluster")
        ax.tick_params(axis="x", rotation=0)

    for ax in axes[len(variables):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# Re-attaching the held-aside outliers and exporting
# ============================================================

def assign_outliers(outlier_df, feature_cols, scaler, kmeans_model):
    """Assign held-aside outliers to their nearest KMeans centroid."""
    Xo = transform_with_scaler(outlier_df, feature_cols, scaler)
    return kmeans_model.predict(Xo)


def build_full_assignment(regular_df, regular_labels,
                          outlier_df=None, outlier_labels=None,
                          label_col="cluster", id_name="customer_id"):
    """Combine regular and outlier labels into one customer-level table."""
    reg = pd.DataFrame({id_name: regular_df.index, label_col: regular_labels})
    parts = [reg]
    if outlier_df is not None and outlier_labels is not None and len(outlier_df):
        out = pd.DataFrame({id_name: outlier_df.index, label_col: outlier_labels})
        parts.append(out)
    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(subset=id_name).sort_values(id_name).reset_index(drop=True)
    return full


# ============================================================
# Scaler comparison (Standard vs MinMax vs Robust)
# ============================================================

def compare_scalers(df, feature_cols, k_range=range(2, 11),
                    random_state=0, n_init=10, sample_size=8000):
    """Mean silhouette per scaler and k."""
    scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(),
               "Robust": RobustScaler()}
    Xraw = df[feature_cols].astype(float)
    rows = []
    for name, scaler in scalers.items():
        Xs = scaler.fit_transform(Xraw)
        for k in k_range:
            labels = KMeans(n_clusters=k, random_state=random_state,
                            n_init=n_init).fit_predict(Xs)
            s = silhouette_score(Xs, labels,
                                 sample_size=min(sample_size, len(labels)),
                                 random_state=random_state)
            rows.append({"scaler": name, "k": k, "silhouette": round(float(s), 4)})
    return pd.DataFrame(rows)


def plot_scaler_comparison(scaler_df, mark_k=None):
    """Plot silhouette-vs-k, one line per scaler."""
    plt.figure(figsize=(9, 5))
    for name, g in scaler_df.groupby("scaler"):
        plt.plot(g["k"], g["silhouette"], marker="o", label=name, color=PROJECT_PALETTE[len(plt.gca().lines) % len(PROJECT_PALETTE)])
    if mark_k is not None:
        plt.axvline(mark_k, color=ACCENT_COLOR, linestyle="--", alpha=0.6,
                    label=f"chosen k={mark_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Mean silhouette")
    plt.title("Scaler comparison (KMeans silhouette)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# Silhouette "blade" plot (per-sample silhouette by cluster)
# ============================================================

def plot_silhouette_blades(X, labels, sample_size=10000, random_state=0,
                           title="Silhouette plot - KMeans"):
    """Draw per-cluster silhouette distributions with the overall average line."""
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.cm as cm
    X = np.asarray(X)
    labels = np.asarray(labels)
    if len(labels) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(labels), sample_size, replace=False)
        X, labels = X[idx], labels[idx]

    avg = silhouette_score(X, labels)
    sample_sil = silhouette_samples(X, labels)
    clusters = sorted(np.unique(labels))

    plt.figure(figsize=(8, 6))
    y_lower = 10
    for ci in clusters:
        vals = np.sort(sample_sil[labels == ci])
        size = len(vals)
        y_upper = y_lower + size
        color = PROJECT_PALETTE[ci % len(PROJECT_PALETTE)]
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                          facecolor=color, edgecolor=color, alpha=0.8)
        plt.text(-0.02, y_lower + size / 2, str(ci))
        y_lower = y_upper + 10
    plt.axvline(avg, color=ACCENT_COLOR, linestyle="--", label=f"Avg = {avg:.2f}")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.yticks([])
    plt.show()
    return avg


# ============================================================
# 2-D embeddings for visual inspection (PCA + UMAP)
# ============================================================

def embed_pca(X, n_components=2, random_state=0):
    """2-D PCA embedding of the scaled clustering matrix."""
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def embed_umap(X, n_neighbors=15, min_dist=0.1, random_state=0):
    """2-D UMAP embedding. Falls back to t-SNE if umap-learn is unavailable."""
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=random_state)
        return reducer.fit_transform(X), "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=random_state,
                   init="pca", perplexity=30).fit_transform(X)
        return emb, "t-SNE (UMAP fallback)"


def embed_tsne(X, perplexity=30, random_state=0):
    """2-D t-SNE embedding for local neighbourhood visual inspection."""
    from sklearn.manifold import TSNE
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(X)


def visualize_dimensionality_reduction(transformation, labels, title="Embedding", cmap_name=None):
    """Plot a 2-D dimensionality reduction coloured by cluster labels."""
    labels = np.asarray(labels).astype(int)
    transformation = np.asarray(transformation)
    unique_labels = np.unique(labels)
    default_cmap = "tab10" if len(unique_labels) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name or default_cmap, len(unique_labels))

    plt.figure(figsize=(9, 7))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            transformation[mask, 0],
            transformation[mask, 1],
            s=8,
            alpha=0.65,
            color=cmap(i),
            label=f"Cluster {label}",
        )

    plt.legend(title="Clusters", fontsize=8, loc="best", frameon=True)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.show()


def plot_embedding(embedding, labels, title="Embedding", method_name=None):
    """Scatter a 2-D embedding coloured by cluster label."""
    final_title = title if not method_name else f"{title} ({method_name})"
    visualize_dimensionality_reduction(embedding, labels, title=final_title)

def plot_umap_label_comparison(X_sample, labels_a, labels_b,
                               title_a="UMAP - KMeans labels",
                               title_b="UMAP - Hierarchical labels",
                               random_state=0):
    """Plot the same UMAP projection twice using two alternative label sets."""
    embedding, method = embed_umap(X_sample, random_state=random_state)
    plot_embedding(embedding, labels_a, title=title_a, method_name=method)
    plot_embedding(embedding, labels_b, title=title_b, method_name=method)
    return embedding, method


# ============================================================
# Feature-set comparison (which columns to cluster on)
# ============================================================

def add_share_features(df):
    """Add category-share columns (category_spend / total_spend) to *df* in-place.

    Each new column is named ``<category>_share`` and represents the fraction of
    lifetime spend allocated to that category.  Customers with zero total spend
    get share = 0 to avoid NaN.  The original spend columns are preserved.

    Returns the list of share column names (excluding groceries).
    """
    spend_cols = [c for c in df.columns if c.startswith("lifetime_spend_")
                  and c != "lifetime_spend_groceries"]
    # Total spend from log column when available, otherwise sum of categories
    if "log_total_spend" in df.columns:
        total = np.expm1(df["log_total_spend"]).replace(0, np.nan)
    else:
        total = df[spend_cols].sum(axis=1).replace(0, np.nan)

    share_cols = []
    for col in spend_cols:
        category = col.replace("lifetime_spend_", "")
        share_name = f"{category}_share"
        df[share_name] = (df[col] / total).fillna(0)
        share_cols.append(share_name)
    return share_cols


def build_candidate_feature_sets(df):
    """Return candidate feature spaces to evaluate for clustering."""
    abss = [c for c in df.columns if c.startswith("lifetime_spend_")]
    abss_no_groceries = [c for c in abss if c != "lifetime_spend_groceries"]
    shares_no_groceries = [c for c in df.columns if c.endswith("_share")
                           and not c.startswith("grocery")]
    eng = [c for c in ["log_total_spend", "distinct_stores_visited",
                       "percentage_of_products_bought_promotion", "tenure",
                       "number_complaints", "lifetime_total_distinct_products"]
           if c in df.columns]
    demo = [c for c in ["customer_age", "education_level", "total_children"]
            if c in df.columns]
    promo = [c for c in ["percentage_of_products_bought_promotion"] if c in df.columns]
    return {
        # ---- value-based: absolute lifetime spend ----
        "lifetime_spend": (abss, False),
        "lifetime_spend no groceries": (abss_no_groceries, False),
        "log_lifetime_spend": (abss, True),
        "log_lifetime_spend no groceries": (abss_no_groceries, True),
        # ---- absolute spend + behaviour ----
        "spend + promo": (abss + promo, False),
        "spend + promo no groceries": (abss_no_groceries + promo, False),
        "log_spend + promo": (abss + promo, True),
        "log_spend + promo no groceries": (abss_no_groceries + promo, True),
        # ---- preference-based: category shares (relative spend) ----
        "shares no groceries": (shares_no_groceries + promo, False),
        # ---- value + demographics / engagement ----
        "log_spend + demo": (abss + demo, True),
        "log_spend + demo no groceries": (abss_no_groceries + demo, True),
        "log_spend + engagement + demo": (abss + eng + demo, True),
        "log_spend + engagement + demo no groceries": (abss_no_groceries + eng + demo, True),
    }


def compare_feature_sets(df, candidate_sets, k=8, scaler=None,
                         random_state=0, n_init=10, sample_size=8000):
    """Silhouette at a fixed k for each candidate feature set (same scaler).

    candidate_sets : dict name -> (columns, log_absolute_spend_flag)
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = scaler or MinMaxScaler()
    rows = []
    for name, (cols, logabs) in candidate_sets.items():
        X = df[cols].astype(float).copy()
        if logabs:
            for c in X.columns:
                if c.startswith("lifetime_spend_"):
                    X[c] = np.log1p(X[c].clip(lower=0))
        Xs = scaler.fit_transform(X)
        labels = KMeans(n_clusters=k, random_state=random_state,
                        n_init=n_init).fit_predict(Xs)
        s = silhouette_score(Xs, labels,
                             sample_size=min(sample_size, len(labels)),
                             random_state=random_state)
        rows.append({"feature_set": name, "n_features": len(cols),
                     "silhouette": round(float(s), 4)})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)


def subsample(X, labels, n=8000, random_state=0):
    """Aligned random subsample of (X, labels) for fast 2-D embedding plots."""
    X = np.asarray(X); labels = np.asarray(labels)
    if len(labels) <= n:
        return X, labels
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(labels), n, replace=False)
    return X[idx], labels[idx]


# ============================================================
# Feature pipelines and model-comparison helpers
# ============================================================

def get_scaler(name):
    """Return a fresh scaler instance by name."""
    if name is None or str(name).lower() == "none":
        return None
    table = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}
    if name not in table:
        raise ValueError(f"Unknown scaler '{name}'. Choose from {list(table) + ['None']}.")
    return table[name]()


def apply_feature_pipeline(df, cols, logabs=False, scaler=None, fit=False):
    """Build the feature matrix, optionally log-transforming spend columns."""
    X = df[cols].astype(float).copy()
    if logabs:
        for c in X.columns:
            if c.startswith("lifetime_spend_"):
                X[c] = np.log1p(X[c].clip(lower=0))
    if scaler is None:
        return X.to_numpy()
    return scaler.fit_transform(X) if fit else scaler.transform(X)




def add_granular_technology_spend(df, raw_customer_info):
    """Add electronics and videogames spend back from the raw customer file."""
    out = df.copy()
    raw = raw_customer_info.copy()
    if "customer_id" in raw.columns:
        raw = raw.set_index("customer_id")
    cols = ["lifetime_spend_electronics", "lifetime_spend_videogames"]
    available = [c for c in cols if c in raw.columns]
    if not available:
        return out
    joined = raw[available].reindex(out.index)
    for col in available:
        out[col] = joined[col]
    return out


def cap_iqr(df, cols, iqr_k=3.0):
    """Apply conservative IQR capping to selected numerical columns."""
    out = df[cols].astype(float).copy()
    if iqr_k is None:
        return out
    for col in out.columns:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            out[col] = out[col].clip(q1 - iqr_k * iqr, q3 + iqr_k * iqr)
    return out


def build_solution_matrix(df, cols, scaler_name="Standard", iqr_k=None,
                          impute_neighbors=5):
    """Create a capped, scaled and imputed matrix for one clustering solution."""
    X = cap_iqr(df, cols, iqr_k=iqr_k)
    scaler = get_scaler(scaler_name)
    X_scaled = X.to_numpy() if scaler is None else scaler.fit_transform(X)
    if np.isnan(X_scaled).any():
        X_scaled = KNNImputer(n_neighbors=impute_neighbors).fit_transform(X_scaled)
    return X_scaled, scaler


def fit_kmeans_solution(df, cols, k=8, scaler_name="Standard", iqr_k=None,
                        random_state=42, n_init=30, sample_size=10000):
    """Fit KMeans and return model, matrix, labels and compact validation metrics."""
    X, scaler = build_solution_matrix(df, cols, scaler_name=scaler_name, iqr_k=iqr_k)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
    labels = model.labels_
    sil = silhouette_score(
        X,
        labels,
        sample_size=min(sample_size, len(labels)),
        random_state=random_state,
    )
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(labels), min(sample_size, len(labels)), replace=False)
    sil_samples = silhouette_samples(X[idx], labels[idx])
    sizes = pd.Series(labels).value_counts().sort_index()
    metrics = {
        "k": int(k),
        "n_features": len(cols),
        "scaler": scaler_name,
        "iqr_k": iqr_k,
        "silhouette": round(float(sil), 4),
        "negative_silhouette_pct": round(float((sil_samples < 0).mean() * 100), 2),
        "min_cluster_pct": round(float((sizes / len(labels) * 100).min()), 2),
        "max_cluster_pct": round(float((sizes / len(labels) * 100).max()), 2),
    }
    return model, X, labels, scaler, metrics




def clustering_metrics(X, labels, sample_size=10000, random_state=42):
    """Compact validation metrics for an already fitted clustering solution."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    sil = silhouette_score(
        X,
        labels,
        sample_size=min(sample_size, len(labels)),
        random_state=random_state,
    )
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(labels), min(sample_size, len(labels)), replace=False)
    sil_values = silhouette_samples(X[idx], labels[idx])
    sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "silhouette": round(float(sil), 4),
        "negative_silhouette_pct": round(float((sil_values < 0).mean() * 100), 2),
        "min_cluster_pct": round(float((sizes / len(labels) * 100).min()), 2),
        "max_cluster_pct": round(float((sizes / len(labels) * 100).max()), 2),
    }

def solution_size_table(labels, names=None):
    """Customer count and percentage per cluster."""
    labels = pd.Series(labels, name="cluster")
    sizes = labels.value_counts().sort_index()
    out = pd.DataFrame({
        "cluster": sizes.index,
        "customers": sizes.values,
        "share_%": (sizes.values / len(labels) * 100).round(2),
    })
    if names:
        out["name"] = out["cluster"].map(names)
    return out


def cluster_preference_table(df, labels, cols, names=None):
    """Min max scaled cluster means for the variables used in one solution."""
    tmp = df.copy()
    tmp["cluster"] = labels
    means = tmp.groupby("cluster")[cols].mean()
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(means),
        index=means.index,
        columns=means.columns,
    )
    if names:
        scaled.index = [f"{i} - {names.get(i, i)}" for i in scaled.index]
    return scaled.round(2)


def plot_cluster_preference_table(pref, title="Cluster preference map"):
    """Heatmap for a cluster preference table returned by cluster_preference_table."""
    labels = {
        "lifetime_spend_electronics": "Electronics",
        "lifetime_spend_videogames": "Videogames",
        "lifetime_spend_vegetables": "Vegetables",
        "lifetime_spend_hygiene": "Hygiene",
        "lifetime_spend_petfood": "Petfood",
        "percentage_of_products_bought_promotion": "Promotions",
        "lifetime_spend_meat": "Meat",
        "lifetime_spend_fish": "Fish",
        "lifetime_spend_technology": "Technology",
        "lifetime_spend_nonalcohol_drinks": "Non alcohol drinks",
        "lifetime_spend_alcohol_drinks": "Alcohol drinks",
    }
    data = pref.rename(columns={c: labels.get(c, c) for c in pref.columns})
    plt.figure(figsize=(max(8, data.shape[1] * 1.25), max(4, data.shape[0] * 0.65)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=sequential_cmap(),
        linewidths=0.5,
        cbar_kws={"shrink": 0.7, "label": "Relative preference"},
    )
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()
    return data


def silhouette_grid(df, candidate_sets, k_range=range(2, 11), scaler_name="Robust",
                    random_state=0, n_init=10, sample_size=8000):
    """Mean silhouette for every (feature_set, k) under one scaler. Tidy DataFrame."""
    scaler_proto = scaler_name
    rows = []
    for fname, (cols, logabs) in candidate_sets.items():
        for k in k_range:
            Xs = apply_feature_pipeline(df, cols, logabs, get_scaler(scaler_proto), fit=True)
            labels = KMeans(n_clusters=k, random_state=random_state,
                            n_init=n_init).fit_predict(Xs)
            s = silhouette_score(Xs, labels,
                                 sample_size=min(sample_size, len(labels)),
                                 random_state=random_state)
            rows.append({"feature_set": fname, "k": k, "silhouette": round(float(s), 4)})
    return pd.DataFrame(rows)


def plot_silhouette_grid(grid_df, title="Silhouette by feature set and k"):
    """Heatmap of the silhouette grid by feature set and k."""
    piv = grid_df.pivot(index="feature_set", columns="k", values="silhouette")
    plt.figure(figsize=(max(8, piv.shape[1]), max(4, piv.shape[0] * 0.6)))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap=sequential_cmap(), linewidths=0.5,
                cbar_kws={"shrink": 0.6})
    plt.title(title)
    plt.ylabel("Feature set")
    plt.xlabel("Number of clusters (k)")
    plt.tight_layout()
    plt.show()
    return piv


# Embedded feature importance, used post-hoc to explain the final labels.

def embedded_feature_importance(X, labels, feature_cols, method="both",
                                C=0.5, n_estimators=300, random_state=0):
    """Embedded-method importance of each clustering feature for the segments.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        The SAME scaled matrix used to fit the clustering (so importances are
        measured in the representation the distance actually used).
    labels : array-like (n_samples,)
        Cluster labels (the pseudo-target).
    feature_cols : list[str]
        Names for the columns of X, in order.
    method : {'both', 'lasso', 'forest'}
    C : float
        Inverse L1 strength for the logistic model (smaller = sparser).
    n_estimators : int
        Trees for the Random Forest.

    Returns
    -------
    pandas.DataFrame indexed by feature with the requested importance columns
    (each normalised to sum to 1), sorted descending. A `lasso_selected`
    boolean column marks features the L1 model kept (non-zero coefficient).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.shape[1] != len(feature_cols):
        raise ValueError("X columns and feature_cols length differ.")

    out = pd.DataFrame(index=pd.Index(feature_cols, name="feature"))

    if method in ("lasso", "both"):
        clf = LogisticRegression(
            penalty="l1", solver="saga", C=C,
            max_iter=2000, random_state=random_state,
        ).fit(X, labels)
        imp = np.abs(clf.coef_).mean(axis=0)        # mean |coef| over OvR classes
        total = imp.sum()
        out["lasso_importance"] = imp / total if total else imp
        out["lasso_selected"] = imp > 0

    if method in ("forest", "both"):
        rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
        ).fit(X, labels)
        out["forest_importance"] = rf.feature_importances_

    sort_col = "lasso_importance" if "lasso_importance" in out else "forest_importance"
    return out.sort_values(sort_col, ascending=False).round(4)


def select_features_embedded(importance_df, column="forest_importance",
                             threshold="median"):
    """Return the features an embedded method would keep.

    threshold : 'median' | 'mean' | float
        Features with importance strictly above the threshold are kept. Use
        this to turn the importance table into an explicit feature shortlist
        that can be compared against the filter (correlation) decision.
    """
    col = importance_df[column].dropna()
    if threshold == "median":
        cut = col.median()
    elif threshold == "mean":
        cut = col.mean()
    else:
        cut = float(threshold)
    return col[col > cut].index.tolist()


def plot_embedded_importance(importance_df, title="Embedded feature importance"):
    """Horizontal bar chart of the embedded importances (one bar group per method)."""
    cols = [c for c in ("lasso_importance", "forest_importance")
            if c in importance_df.columns]
    data = importance_df[cols].sort_values(cols[0])
    ax = data.plot(kind="barh", figsize=(9, max(4, len(data) * 0.45)),
                   color=[MAIN_COLOR, SECONDARY_COLOR][:len(cols)])
    ax.set_xlabel("Normalised importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Standardised profile views
# ============================================================

def plot_profile_heatmap_z(profile_df, title="Cluster profile (standardised per feature)"):
    """Heatmap of a profile table with each feature standardised across clusters.

    Pass the SAME table you give to plot_profile_heatmap (it may include the
    OVERALL row; it is dropped before plotting). Diverging colours are centred
    at 0 = overall average.
    """
    data = profile_df.drop(index="OVERALL", errors="ignore").astype(float)
    z = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, np.nan)
    z = z.fillna(0.0)
    plt.figure(figsize=(max(8, z.shape[1] * 0.9), max(4, z.shape[0] * 0.7)))
    sns.heatmap(z, annot=True, fmt="+.1f", cmap=diverging_cmap(), center=0,
                linewidths=0.5, cbar_kws={"shrink": 0.6, "label": "std devs from overall"})
    plt.title(title)
    plt.ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return z.round(2)


# ============================================================
# Granular feature-set search
# ============================================================

def build_granular_feature_sets(df):
    """Curated, granular candidate sets (one behaviour/lifecycle block at a time).

    Each entry is (columns, log_absolute_spend?), matching silhouette_grid /
    plot_silhouette_grid. Identity/geo (is_male, loyalty, lat, long) are left
    out of every set on purpose: they are profiling variables, not distance
    drivers, and empirically carry ~0 importance for separating segments.
    """
    spend = [c for c in df.columns if c.startswith("lifetime_spend_")]
    spend_ng = [c for c in spend if c != "lifetime_spend_groceries"]
    promo = [c for c in ["percentage_of_products_bought_promotion"] if c in df.columns]
    engage = [c for c in ["distinct_stores_visited", "lifetime_total_distinct_products"]
              if c in df.columns]
    family = [c for c in ["total_children"] if c in df.columns]

    sets = {
        "spend": (spend, True),
        "spend no groceries": (spend_ng, True),
        "spend_ng + promo": (spend_ng + promo, True),
        "spend_ng + promo + engagement": (spend_ng + promo + engage, True),
        "spend_ng + promo + family": (spend_ng + promo + family, True),
        "spend_ng + promo + engagement + family": (spend_ng + promo + engage + family, True),
    }
    return {k: (cols, log) for k, (cols, log) in sets.items() if cols}


# ============================================================
# Self-Organising Map benchmark
# ============================================================

def som_quantization_curve(X, grid=(3, 3), iterations=1000, checkpoints=None,
                           sigma=0.5, learning_rate=1.0, sample_size=8000,
                           random_state=0):
    """Train a SOM in blocks and return its quantization-error curve."""
    from minisom import MiniSom

    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X_train = X[idx]
    else:
        X_train = X

    checkpoints = checkpoints or [50, 100, 200, 400, 700, iterations]
    checkpoints = sorted({int(c) for c in checkpoints if 0 < int(c) <= iterations})
    if checkpoints[-1] != iterations:
        checkpoints.append(iterations)

    som = MiniSom(
        grid[0], grid[1], input_len=X.shape[1], sigma=sigma,
        learning_rate=learning_rate, neighborhood_function="gaussian",
        random_seed=random_state,
    )
    som.random_weights_init(X_train)

    rows = []
    previous = 0
    for step in checkpoints:
        som.train_random(X_train, step - previous, verbose=False)
        previous = step
        rows.append({
            "iteration": step,
            "quantization_error": round(float(som.quantization_error(X_train)), 4),
        })
    return pd.DataFrame(rows), som


def assign_som_units(som, X):
    """Assign each row to its best-matching SOM unit."""
    X = np.asarray(X, dtype=float)
    weights = som.get_weights()
    n_cols = weights.shape[1]
    labels = []
    for row in X:
        r, c = som.winner(row)
        labels.append(r * n_cols + c)
    return np.asarray(labels, dtype=int)


def plot_som_quantization_curve(curve_df):
    """Plot SOM quantization error across training iterations."""
    plt.figure(figsize=(8, 4))
    plt.plot(curve_df["iteration"], curve_df["quantization_error"], marker="o", color=MAIN_COLOR)
    plt.xlabel("Training iterations")
    plt.ylabel("Quantization error")
    plt.title("SOM quantization error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_som_umatrix(som, title="SOM U-Matrix", annot=False):
    """Plot the SOM distance map."""
    distances = som.distance_map().T
    height = max(4, distances.shape[0] * 0.35)
    width = max(5, distances.shape[1] * 0.35)

    plt.figure(figsize=(width, height))
    sns.heatmap(
        distances,
        cmap=sequential_cmap(),
        annot=annot,
        fmt=".2f",
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Neighbour distance"},
    )
    plt.title(title)
    plt.xlabel("SOM row")
    plt.ylabel("SOM column")
    plt.tight_layout()
    plt.show()


def plot_som_unit_counts(labels, grid=(3, 3), title="Customers per SOM unit", annot=None):
    """Plot the number of observations assigned to each SOM unit."""
    counts = pd.Series(labels).value_counts().sort_index()
    mat = np.zeros(grid[0] * grid[1], dtype=int)
    for unit, count in counts.items():
        if 0 <= int(unit) < len(mat):
            mat[int(unit)] = int(count)
    mat = mat.reshape(grid)

    if annot is None:
        annot = max(grid) <= 10

    height = max(4, grid[1] * 0.35)
    width = max(5, grid[0] * 0.35)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        mat.T,
        cmap=sequential_cmap(),
        annot=annot,
        fmt="d",
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Customers"},
    )
    plt.title(title)
    plt.xlabel("SOM row")
    plt.ylabel("SOM column")
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(mat, index=[f"row_{i}" for i in range(grid[0])],
                        columns=[f"col_{j}" for j in range(grid[1])])


def plot_som_feature_maps(som, feature_names, features=None, n_cols=3,
                          cmap=diverging_cmap()):
    """Plot SOM weight maps for selected features."""
    weights = som.get_weights()
    feature_names = list(feature_names)
    features = features or feature_names
    selected = [f for f in features if f in feature_names]
    if not selected:
        raise ValueError("None of the requested features exists in feature_names.")

    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, selected):
        idx = feature_names.index(feature)
        sns.heatmap(
            weights[:, :, idx].T,
            ax=ax,
            cmap=cmap,
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(feature)
        ax.set_xlabel("SOM row")
        ax.set_ylabel("SOM column")

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# Ensemble / consensus clustering
# ============================================================

def consensus_kmeans(X, k, n_runs=25, random_state=0, n_init=5):
    """Stability-based ensemble of KMeans runs.

    Returns
    -------
    (consensus_labels, stability)
        consensus_labels : majority-vote segment per row (aligned label space)
        stability        : fraction of runs (0-1) agreeing with the consensus
    """
    from scipy.optimize import linear_sum_assignment

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    runs = [KMeans(n_clusters=k, random_state=random_state + r, n_init=n_init)
            .fit_predict(X) for r in range(n_runs)]

    ref = runs[0]
    aligned = [ref]
    for lab in runs[1:]:
        cm = confusion_matrix(ref, lab, labels=list(range(k)))
        row, col = linear_sum_assignment(-cm)
        mapping = {c: r for r, c in zip(row, col)}
        aligned.append(np.array([mapping.get(x, x) for x in lab]))

    A = np.vstack(aligned).T
    counts = np.zeros((n, k), dtype=int)
    for j in range(A.shape[1]):
        counts[np.arange(n), A[:, j]] += 1
    consensus = counts.argmax(axis=1)
    stability = counts.max(axis=1) / A.shape[1]
    return consensus, stability


def plot_stability(stability, title="Consensus stability per customer"):
    """Histogram of the per-customer ensemble agreement scores."""
    stability = np.asarray(stability)
    plt.figure(figsize=(8, 4))
    plt.hist(stability, bins=20, color=MAIN_COLOR, edgecolor=LIGHT_COLOR)
    plt.axvline(stability.mean(), color=ACCENT_COLOR, linestyle="--",
                label=f"mean = {stability.mean():.3f}")
    plt.xlabel("Fraction of runs agreeing with the consensus label")
    plt.ylabel("Customers")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Notebook workflow helpers
# ============================================================

def run_method_benchmarks(X, k_range=range(2, 11), random_state=0):
    """Run KMeans, agglomerative and centroid Ward benchmark tables."""
    kmeans_search = kmeans_k_benchmark(
        X, k_range=k_range, random_state=random_state, n_init=30, sample_size=8000
    )
    hierarchical_search = hierarchical_k_benchmark(
        X,
        k_range=k_range,
        linkages=["ward", "complete", "average", "single"],
        sample_size=5000,
        random_state=random_state,
    )
    centroid_ward_search = centroid_ward_k_benchmark(
        X,
        macro_k_range=k_range,
        micro_k=20,
        random_state=random_state,
        n_init=30,
        sample_size=8000,
    )
    plot_df = pd.concat(
        [
            kmeans_search[["method", "k", "silhouette"]],
            hierarchical_search[["method", "k", "silhouette"]],
            centroid_ward_search[["method", "k", "silhouette"]],
        ],
        ignore_index=True,
    )
    summary = benchmark_summary_table(kmeans_search, hierarchical_search, centroid_ward_search)
    return {
        "plot": plot_df,
        "summary": summary,
        "kmeans": kmeans_search,
        "hierarchical": hierarchical_search,
        "centroid_ward": centroid_ward_search,
    }


def compare_hierarchical_linkages(X, k, final_labels, linkages=None,
                                  sample_size=5000, random_state=0):
    """Compare final KMeans labels with agglomerative labels on one sample."""
    linkages = linkages or ["ward", "complete", "average", "single"]
    rows = []
    comparisons = {}
    for linkage_name in linkages:
        sample_idx, labels = fit_hierarchical_sample(
            X, k, sample_size=sample_size, linkage=linkage_name, random_state=random_state
        )
        sil = silhouette_score(X[sample_idx], labels)
        rows.append({
            "method": f"Agglomerative-{linkage_name}",
            "sample_size": len(sample_idx),
            "k": int(k),
            "silhouette_sample": round(float(sil), 4),
        })
        comparisons[linkage_name] = compare_solutions(
            final_labels[sample_idx], labels,
            name_a="KMeans", name_b=linkage_name.title(),
        )
    return pd.DataFrame(rows), comparisons


def benchmark_label_tracking():
    """Document where benchmark labels are stored in the workflow."""
    return pd.DataFrame([
        {
            "solution": "Final KMeans",
            "label_column": "cluster",
            "scope": "regular customer dataset",
            "purpose": "final assignment before outlier reintegration",
        },
        {
            "solution": "Agglomerative linkages",
            "label_column": "hierarchical comparison tables",
            "scope": "sample only",
            "purpose": "benchmark against KMeans under several linkages",
        },
        {
            "solution": "Centroid Ward macro clusters",
            "label_column": "macro_cluster",
            "scope": "regular customer dataset",
            "purpose": "two stage hierarchical benchmark",
        },
        {
            "solution": "DBSCAN",
            "label_column": "dbscan_cluster if enabled",
            "scope": "benchmark only",
            "purpose": "density and noise sensitivity check",
        },
        {
            "solution": "SOM",
            "label_column": "som_unit",
            "scope": "regular customer dataset",
            "purpose": "topology based visual profiling",
        },
    ])


def fit_centroid_ward_macro(X, feature_cols, k, micro_k=20, random_state=0,
                            n_init=20, plot=True):
    """Fit KMeans micro clusters and merge their centroids with Ward linkage."""
    from scipy.cluster.hierarchy import linkage, fcluster

    micro = fit_kmeans(X, micro_k, random_state=random_state, n_init=n_init)
    centroids = pd.DataFrame(
        micro.cluster_centers_,
        columns=feature_cols,
        index=[f"C{i}" for i in range(micro_k)],
    )
    Z = linkage(centroids.values, method="ward")
    if plot:
        plt.figure(figsize=(10, 5))
        dendrogram(
            Z,
            labels=centroids.index.tolist(),
            leaf_rotation=90,
            leaf_font_size=10,
        )
        plt.title("Hierarchical clustering on KMeans centroids (Ward linkage)")
        plt.ylabel("Ward distance")
        plt.tight_layout()
        plt.show()

    centroid_labels = fcluster(Z, t=int(k), criterion="maxclust") - 1
    mapping = dict(zip(range(micro_k), centroid_labels))
    macro_labels = np.array([mapping[x] for x in micro.labels_])
    return micro, centroids, Z, macro_labels


def centroid_ward_report(X, final_labels, macro_labels):
    """Return compact metrics for the centroid Ward benchmark."""
    return pd.DataFrame([
        {
            "solution": "Final KMeans",
            "silhouette": round(float(silhouette_score(X, final_labels)), 4),
        },
        {
            "solution": "Centroid Ward macro",
            "silhouette": round(float(silhouette_score(X, macro_labels)), 4),
        },
    ])


def plot_macro_embeddings(X, macro_labels, sample_size=8000, random_state=0):
    """Plot PCA and UMAP views for macro cluster labels."""
    X_sub, labels_sub = subsample(X, macro_labels, n=sample_size, random_state=random_state)
    plot_embedding(
        embed_pca(X_sub),
        labels_sub,
        title="PCA - Centroid Ward macro clusters",
        method_name="PCA",
    )
    emb, method = embed_umap(X_sub, random_state=random_state)
    plot_embedding(
        emb,
        labels_sub,
        title="UMAP - Centroid Ward macro clusters",
        method_name=method,
    )


def dbscan_benchmark_table(X, eps_values=None, min_samples_values=None,
                           sample_size=8000, random_state=0):
    """Run DBSCAN grid and return valid and non valid parameter tables."""
    results = dbscan_grid(
        X,
        eps_values=eps_values,
        min_samples_values=min_samples_values,
        sample_size=sample_size,
        random_state=random_state,
    )
    valid = valid_dbscan_results(results)
    invalid = results[results["silhouette_non_noise"].isna()][
        ["eps", "min_samples", "n_clusters", "noise_pct"]
    ].reset_index(drop=True)
    return results, valid, invalid

def practical_dbscan_candidates(results, min_clusters=2, max_clusters=12, max_noise_pct=35):
    """Return DBSCAN candidates with valid silhouette and manageable noise."""
    out = results.dropna(subset=["silhouette_non_noise"]).copy()
    out = out[
        out["n_clusters"].between(min_clusters, max_clusters)
        & (out["noise_pct"] <= max_noise_pct)
    ]
    return out.sort_values(
        ["silhouette_non_noise", "noise_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)


def maybe_fit_dbscan(X, run=False, eps=0.9, min_samples=10):
    """Optionally fit DBSCAN; otherwise keep it as benchmark only."""
    if not run:
        print("DBSCAN is kept as a benchmark only. The final assignment is not based on DBSCAN labels.")
        return None
    labels = fit_dbscan(X, eps=float(eps), min_samples=int(min_samples))
    non_noise = labels != -1
    print("Selected DBSCAN parameters:")
    print("eps:", eps, "| min_samples:", min_samples)
    print("Clusters excluding noise:", len(set(labels[non_noise])))
    print("Noise %:", round((labels == -1).mean() * 100, 2))
    display(pd.Series(labels).value_counts().sort_index())
    return labels


def build_som_features(df):
    """Default profiling variables used in the SOM diagnostic."""
    return [c for c in [
        "lifetime_spend_groceries",
        "lifetime_spend_vegetables",
        "lifetime_spend_meat",
        "lifetime_spend_fish",
        "lifetime_spend_alcohol_drinks",
        "lifetime_spend_hygiene",
        "lifetime_spend_petfood",
        "lifetime_spend_technology",
        "percentage_of_products_bought_promotion",
        "total_children",
        "customer_age",
        "education_level",
        "number_complaints",
    ] if c in df.columns]


def som_grid_search(
    X,
    sigmas=(0.5, 1.0, 1.5, 2.0),
    learning_rates=(0.3, 0.5, 0.7, 0.9),
    grid=(12, 12),
    iterations=1000,
    sample_size=8000,
    random_state=0,
):
    """Grid search over SOM sigma and learning_rate.

    Returns a DataFrame sorted by quantization_error (ascending), so the
    first row is the best configuration.
    """
    from minisom import MiniSom

    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), sample_size, replace=False)
        X_train = X[idx]
    else:
        X_train = X

    rows = []
    for sigma in sigmas:
        for lr in learning_rates:
            som = MiniSom(
                grid[0], grid[1], input_len=X.shape[1],
                sigma=sigma, learning_rate=lr,
                neighborhood_function="gaussian",
                random_seed=random_state,
            )
            som.random_weights_init(X_train)
            som.train_random(X_train, iterations, verbose=False)
            qe = float(som.quantization_error(X_train))
            te = float(som.topographic_error(X_train))
            rows.append({
                "sigma": sigma,
                "learning_rate": lr,
                "quantization_error": round(qe, 4),
                "topographic_error": round(te, 4),
                "combined_score": round(qe + te, 4),
            })

    results = pd.DataFrame(rows).sort_values("combined_score").reset_index(drop=True)
    return results


def som_kmeans_init(X_som, som, k, n_init=10, random_state=0):
    """Use SOM weight vectors to initialise K-Means centroids.

    The SOM weight matrix is itself clustered with K-Means (n_init=10 is
    enough here because the initialisation space is much smaller). The
    resulting centroids are then used as 'init' for the final K-Means run on
    the full feature matrix X_som, which avoids poor random starts.

    Returns
    -------
    kmeans : fitted KMeans on X_som with SOM-warm-started centroids
    """
    weights = som.get_weights().reshape(-1, X_som.shape[1])
    # cluster the SOM units to get k representative weight vectors
    pre = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(weights)
    # use those centroids as init for the full run
    kmeans = KMeans(n_clusters=k, init=pre.cluster_centers_, n_init=1,
                    random_state=random_state).fit(X_som)
    return kmeans


def run_som_diagnostic(df, scaler_name="MinMax", grid=(12, 12), iterations=1000,
                       sample_size=12000, random_state=0,
                       run_grid_search=True,
                       sigmas=(0.5, 1.0, 1.5, 2.0),
                       learning_rates=(0.3, 0.5, 0.7, 0.9)):
    """Train the SOM diagnostic and plot its main views.

    If run_grid_search=True, a sigma × learning_rate grid search is run first
    and the best configuration (lowest combined QE+TE score) is used for the
    final SOM. Set run_grid_search=False to skip the search and use the
    defaults (sigma=1.2, learning_rate=0.5).
    """
    feature_cols = build_som_features(df)
    scaler = get_scaler(scaler_name)
    X_som = apply_feature_pipeline(df, feature_cols, logabs=True, scaler=scaler, fit=True)

    best_sigma, best_lr = 1.2, 0.5
    if run_grid_search:
        print("Running SOM hyperparameter grid search …")
        gs_results = som_grid_search(
            X_som,
            sigmas=sigmas,
            learning_rates=learning_rates,
            grid=grid,
            iterations=iterations,
            sample_size=min(sample_size, 8000),
            random_state=random_state,
        )
        display(gs_results.head(10))
        best_sigma = float(gs_results.iloc[0]["sigma"])
        best_lr = float(gs_results.iloc[0]["learning_rate"])
        print(f"Best params → sigma={best_sigma}, learning_rate={best_lr}")

    curve, som = som_quantization_curve(
        X_som,
        grid=grid,
        iterations=iterations,
        checkpoints=[50, 100, 200, 400, 700, iterations],
        sigma=best_sigma,
        learning_rate=best_lr,
        sample_size=sample_size,
        random_state=random_state,
    )
    display(curve)
    plot_som_quantization_curve(curve)
    plot_som_umatrix(som, title="SOM U-Matrix")
    return feature_cols, X_som, curve, som


def assign_and_plot_som(df, som, X_som, feature_cols, grid=(12, 12)):
    """Assign SOM units and plot hit map and feature maps."""
    units = assign_som_units(som, X_som)
    counts = pd.Series(units).value_counts()
    print("Active SOM units:", counts.shape[0], "out of", grid[0] * grid[1])
    display(counts.rename("customers").sort_values(ascending=False).head(15).to_frame())
    plot_som_unit_counts(units, grid=grid, title="SOM hit map", annot=False)
    plot_som_feature_maps(som, feature_names=feature_cols, features=feature_cols, n_cols=3)
    return units


def consensus_report(X, labels, k, n_runs=25, random_state=0):
    """Run consensus KMeans and print the main stability diagnostics."""
    from sklearn.metrics import adjusted_rand_score

    consensus_labels, stability = consensus_kmeans(
        X, k, n_runs=n_runs, random_state=random_state
    )
    print("Single-run silhouette  :", round(silhouette_score(
        X, labels, sample_size=8000, random_state=random_state), 4))
    print("Consensus  silhouette  :", round(silhouette_score(
        X, consensus_labels, sample_size=8000, random_state=random_state), 4))
    print("Agreement single vs consensus (ARI):",
          round(adjusted_rand_score(labels, consensus_labels), 4))
    print("Mean stability:", round(stability.mean(), 3),
          "| very stable (>=0.9):", f"{(stability >= 0.9).mean() * 100:.1f}%",
          "| ambiguous (<0.6):", f"{(stability < 0.6).mean() * 100:.1f}%")
    plot_stability(stability)
    return consensus_labels, stability


def default_profile_columns(df):
    """Default behavioural and demographic columns used for profiling."""
    return [c for c in [
        "log_total_spend",
        "percentage_of_products_bought_promotion",
        "distinct_stores_visited",
        "lifetime_total_distinct_products",
        "customer_age",
        "education_level",
        "tenure",
        "total_children",
        "number_complaints",
    ] if c in df.columns]


def plot_segment_separation(data, label_col="cluster"):
    """Plot scaled means and boxplots for the main profiling variables."""
    cols = default_profile_columns(data)
    profile = profile_clusters(data, label_col, cols)
    scaled = plot_cluster_mean_comparison(profile, title="Normalised cluster mean comparison")
    display(scaled)
    box_cols = [c for c in [
        "log_total_spend",
        "percentage_of_products_bought_promotion",
        "distinct_stores_visited",
        "customer_age",
        "total_children",
        "number_complaints",
    ] if c in data.columns]
    plot_boxplot_grid(data, box_cols, label_col=label_col)
    return scaled


def geographic_profile(data, label_col="cluster"):
    """Return and plot location profiles by segment."""
    if not {"latitude", "longitude"}.issubset(data.columns):
        return pd.DataFrame()
    profile = (
        data
        .groupby(label_col)
        .agg(
            customers=(label_col, "size"),
            mean_latitude=("latitude", "mean"),
            mean_longitude=("longitude", "mean"),
            median_latitude=("latitude", "median"),
            median_longitude=("longitude", "median"),
            latitude_std=("latitude", "std"),
            longitude_std=("longitude", "std"),
        )
        .round(5)
    )
    display(profile)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    clusters = sorted(data[label_col].unique())
    for ax, cluster_id in zip(axes, clusters):
        sub = data[data[label_col] == cluster_id]
        ax.scatter(sub["longitude"], sub["latitude"], s=5, alpha=0.35)
        ax.scatter(
            sub["longitude"].mean(),
            sub["latitude"].mean(),
            s=80,
            color=ACCENT_COLOR,
            marker="x",
        )
        ax.set_title(f"Cluster {cluster_id}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(clusters):]:
        ax.axis("off")
    fig.suptitle("Customer locations by segment (small multiples)", y=1.02)
    plt.tight_layout()
    plt.show()
    return profile


def reattach_outliers_and_export(regular_df, outlier_df, kmeans_model, feature_cols,
                                 logabs, scaler, data_dir, spend_profile=None,
                                 complaints_profile=None):
    """Assign held-aside outliers and save final clustering exports."""
    X_out = apply_feature_pipeline(outlier_df, feature_cols, logabs, scaler, fit=False)
    outlier_labels = kmeans_model.predict(X_out)
    print("Outliers assigned:", len(outlier_labels))

    segments = build_full_assignment(
        regular_df,
        kmeans_model.labels_,
        outlier_df,
        outlier_labels,
        label_col="cluster",
        id_name="customer_id",
    )
    expected = len(regular_df) + len(outlier_df)
    assert segments["customer_id"].nunique() == expected, "Missing customers!"
    print("Customers in final file:", segments["customer_id"].nunique())
    print(segments["cluster"].value_counts().sort_index())

    data_dir = str(data_dir)
    segments.to_csv(f"{data_dir}/customer_segments.csv", index=False)
    cluster_sizes(regular_df, "cluster").to_csv(f"{data_dir}/segment_summary.csv")
    if spend_profile is not None:
        spend_profile.to_csv(f"{data_dir}/segment_spend_profile.csv")
    if complaints_profile is not None:
        complaints_profile.to_csv(f"{data_dir}/segment_complaints_profile.csv")
    print("Saved customer_segments.csv (+ supporting profiles) to", data_dir)
    return segments


def outliers_as_own_cluster(regular_df, outlier_df, kmeans_model, feature_cols,
                             logabs, scaler, data_dir, spend_profile=None,
                             complaints_profile=None):
    """Label held-aside outliers as their own dedicated cluster (cluster K).

    Instead of assigning outliers to their nearest regular centroid, this
    function gives them a single distinct label equal to the number of
    regular clusters (e.g., k=8 → outliers become cluster 8).  This
    preserves the identity of the high-spend outlier segment and makes
    it visible in downstream profiling as a small but meaningful group.

    Returns
    -------
    segments : pd.DataFrame
        One row per customer with columns [customer_id, cluster].
    """
    k = len(np.unique(kmeans_model.labels_))
    outlier_label = k  # cluster index just beyond the regular labels

    # Regular customers keep their fitted labels unchanged
    regular_ids = regular_df.index.to_series().reset_index(drop=True)
    reg_labels  = pd.Series(kmeans_model.labels_, name="cluster")

    # Outliers all receive label `k`
    outlier_ids    = outlier_df.index.to_series().reset_index(drop=True)
    outlier_labels = pd.Series(np.full(len(outlier_df), outlier_label, dtype=int),
                                name="cluster")

    reg_df  = pd.DataFrame({"customer_id": regular_ids, "cluster": reg_labels})
    out_df  = pd.DataFrame({"customer_id": outlier_ids, "cluster": outlier_labels})
    segments = pd.concat([reg_df, out_df], ignore_index=True)

    expected = len(regular_df) + len(outlier_df)
    assert segments["customer_id"].nunique() == expected, "Missing customers!"

    total = len(segments)
    out_n = len(outlier_df)
    print(f"Regular clusters  : {k}  (labels 0–{k-1})")
    print(f"Outlier cluster   : label {outlier_label}  "
          f"n={out_n}  ({out_n/total*100:.2f}% of all customers)")
    print(segments["cluster"].value_counts().sort_index())

    data_dir = str(data_dir)
    segments.to_csv(f"{data_dir}/customer_segments.csv", index=False)

    # Summary includes ALL clusters (regular + outlier)
    all_df = pd.concat([regular_df.copy(), outlier_df.copy()], axis=0)
    all_df["cluster"] = segments.set_index("customer_id")["cluster"]
    cluster_sizes(all_df, "cluster").to_csv(f"{data_dir}/segment_summary.csv")

    if spend_profile is not None:
        spend_profile.to_csv(f"{data_dir}/segment_spend_profile.csv")
    if complaints_profile is not None:
        complaints_profile.to_csv(f"{data_dir}/segment_complaints_profile.csv")

    print("Saved customer_segments.csv (+ supporting profiles) to", data_dir)
    return segments



def plot_sample_dendrogram(X, title, linkage="ward", sample_size=3000,
                           cut_height=None, random_state=0):
    """Fit and plot one truncated dendrogram on a customer sample."""
    _, model = fit_dendrogram_model(
        X, sample_size=sample_size, linkage=linkage, random_state=random_state
    )
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plot_dendrogram(model, truncate_mode="level", p=5)
    if cut_height is not None:
        plt.axhline(
            y=cut_height,
            color=ACCENT_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=f"cut = {cut_height}",
        )
        plt.legend(loc="upper right")
    plt.xlabel("Sample of customers")
    plt.ylabel("Distance")
    plt.show()


def plot_alternative_dendrograms(X, title_suffix, linkages=None, cut_heights=None,
                                 sample_size=3000, random_state=0):
    """Plot complete, average and single linkage dendrogram checks."""
    linkages = linkages or ["complete", "average", "single"]
    cut_heights = cut_heights or {}
    for linkage_name in linkages:
        plot_sample_dendrogram(
            X,
            title=f"{linkage_name.title()} dendrogram (sample) - {title_suffix}",
            linkage=linkage_name,
            sample_size=sample_size,
            cut_height=cut_heights.get(linkage_name),
            random_state=random_state,
        )


def display_method_benchmarks(method_benchmarks):
    """Display the benchmark plot and compact ranking tables."""
    plot_method_k_benchmark(
        method_benchmarks["plot"],
        title="Independent method search: silhouette by k",
    )
    display(method_benchmarks["summary"])

    print("KMeans candidates")
    display(method_benchmarks["kmeans"].sort_values("silhouette", ascending=False).head(5))

    print("Best hierarchical candidates")
    display(method_benchmarks["hierarchical"].head(5))

    print("Best centroid Ward candidates")
    display(method_benchmarks["centroid_ward"].sort_values(
        ["silhouette", "r2"], ascending=[False, False]
    ).head(5))


# ─────────────────────────────────────────────────────────────────────────────
# Consensus KMeans (ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def consensus_kmeans(X, k, n_runs=40, sample_size=8000, linkage="average",
                     random_state=0):
    """Consensus clustering via ensemble of KMeans runs.

    Builds a co-occurrence matrix on a random sample: entry (i,j) = fraction
    of runs where customers i and j ended up in the same cluster.  Then
    converts to a distance matrix and applies AgglomerativeClustering to find
    ``k`` stable groups.  Finally, assigns every point in the full dataset to
    its nearest centroid.

    Parameters
    ----------
    X            : scaled feature matrix (n_samples × n_features)
    k            : number of consensus clusters
    n_runs       : number of KMeans runs (default 40)
    sample_size  : size of the sub-sample used for the co-occurrence matrix
    linkage      : linkage for AgglomerativeClustering on the distance matrix
    random_state : seed for reproducibility

    Returns
    -------
    labels       : cluster label for every row of X  (shape n_samples,)
    comat        : co-occurrence matrix on the sample (shape sample_size × sample_size)
    sample_idx   : indices of the rows used in the sample
    """
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import cdist

    rng = np.random.RandomState(random_state)
    n   = len(X)

    if sample_size >= n:
        sample_idx = np.arange(n)
    else:
        sample_idx = rng.choice(n, sample_size, replace=False)
    Xs = X[sample_idx]
    m  = len(sample_idx)

    comat = np.zeros((m, m), dtype=np.float32)
    for i in range(n_runs):
        labs = KMeans(n_clusters=k, n_init=1,
                      random_state=random_state + i).fit_predict(Xs)
        for c in np.unique(labs):
            members = np.where(labs == c)[0]
            comat[np.ix_(members, members)] += 1.0
    comat /= n_runs

    dist = (1.0 - comat).astype(np.float64)
    np.fill_diagonal(dist, 0.0)

    agg      = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                       linkage=linkage)
    slabels  = agg.fit_predict(dist)

    # Centroids in the original scaled space
    centroids = np.array([Xs[slabels == c].mean(axis=0) for c in range(k)])

    # Assign full dataset to nearest centroid
    labels = cdist(X, centroids).argmin(axis=1)
    return labels, comat, sample_idx


def plot_consensus_heatmap(comat, sample_labels, title="Consensus matrix",
                           max_display=500):
    """Plot sorted co-occurrence heatmap for a consensus run."""
    order = np.argsort(sample_labels)
    sub   = comat[order][:, order]
    if max_display < len(sub):
        step  = len(sub) // max_display
        sub   = sub[::step][:, ::step]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sub, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Co-occurrence rate")
    ax.set_title(title)
    ax.set_xlabel("Customers (sorted by cluster)")
    ax.set_ylabel("Customers (sorted by cluster)")
    plt.tight_layout()
    plt.show()


def consensus_cluster_profile(full_df, labels, feature_cols, demographic_cols=None):
    """Return a z-score profile DataFrame for each consensus cluster."""
    df = full_df.copy()
    df["cluster"] = labels
    pres = [c for c in feature_cols if c in df.columns]
    gm   = df[pres].mean()
    gs   = df[pres].std().replace(0, np.nan)
    rows = {}
    for c in sorted(np.unique(labels)):
        mask = df["cluster"] == c
        n    = mask.sum()
        z    = (df.loc[mask, pres].mean() - gm) / gs
        rows[c] = {"n": n, "pct": round(n / len(df) * 100, 1), **z.round(3).to_dict()}
    return pd.DataFrame(rows).T


def apply_centroid_ward_macro(data, X, feature_cols, k, micro_k=20, random_state=0):
    """Fit centroid Ward macro clusters, attach labels and display diagnostics."""
    micro, centroids, Z, macro_labels = fit_centroid_ward_macro(
        X,
        feature_cols,
        k,
        micro_k=micro_k,
        random_state=random_state,
        n_init=20,
        plot=True,
    )
    data["micro_cluster"] = micro.labels_
    data["macro_cluster"] = macro_labels

    print("Micro-cluster sizes")
    display(data["micro_cluster"].value_counts().sort_index())

    print("Macro-cluster sizes")
    display(data["macro_cluster"].value_counts().sort_index())

    display(centroid_ward_report(X, data["cluster"].values, data["macro_cluster"].values))
    return micro, centroids, Z


# ---------------------------------------------------------------------------
# Outlier strategy comparison: separation vs IQR capping
# ---------------------------------------------------------------------------

def compare_outlier_strategies(
    separation_df,
    capped_df,
    feature_cols,
    k,
    scaler_name="MinMax",
    logabs=False,
    random_state=0,
    sample_size=10_000,
):
    """Compare clustering quality for two outlier handling strategies.

    Parameters
    ----------
    separation_df : pd.DataFrame
        Regular customers only (outliers already removed). The model is
        fitted on this set; outliers are not reattached here — this function
        focuses on clustering quality, not full coverage.
    capped_df : pd.DataFrame
        Full dataset with IQR-capped values (all customers included).
    feature_cols : list[str]
        Columns used for clustering distance.
    k : int
        Number of clusters.
    scaler_name : str
        Scaler to apply ('MinMax', 'Standard', 'Robust').
    logabs : bool
        Whether to apply log1p transform before scaling.
    random_state : int
    sample_size : int
        Silhouette sample size.

    Returns
    -------
    pd.DataFrame
        One row per strategy with silhouette, balance and size metrics.
    """
    results = []

    for label, df in [("Separation (regular only)", separation_df),
                      ("IQR capping (full dataset)", capped_df)]:
        cols = [c for c in feature_cols if c in df.columns]
        scaler = get_scaler(scaler_name)
        X = apply_feature_pipeline(df, cols, logabs, scaler, fit=True)
        model = fit_kmeans(X, k, random_state=random_state)
        labels = model.labels_
        sil = silhouette_score(
            X, labels,
            sample_size=min(sample_size, len(labels)),
            random_state=random_state,
        )
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(labels), min(sample_size, len(labels)), replace=False)
        sil_vals = silhouette_samples(X[idx], labels[idx])
        sizes = pd.Series(labels).value_counts()
        results.append({
            "strategy": label,
            "n_customers": len(df),
            "k": k,
            "scaler": scaler_name,
            "silhouette": round(float(sil), 4),
            "negative_silhouette_pct": round(float((sil_vals < 0).mean() * 100), 2),
            "min_cluster_pct": round(float((sizes / len(labels) * 100).min()), 2),
            "max_cluster_pct": round(float((sizes / len(labels) * 100).max()), 2),
        })

    return pd.DataFrame(results)


def plot_strategy_umap_comparison(
    separation_df,
    capped_df,
    feature_cols,
    k,
    scaler_name="MinMax",
    logabs=False,
    sample_size=6_000,
    random_state=0,
):
    """Side-by-side UMAP plots for separation vs capping strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (label, df) in zip(axes, [
        ("Separation (regular only)", separation_df),
        ("IQR capping (full dataset)", capped_df),
    ]):
        cols = [c for c in feature_cols if c in df.columns]
        scaler = get_scaler(scaler_name)
        X = apply_feature_pipeline(df, cols, logabs, scaler, fit=True)
        model = fit_kmeans(X, k, random_state=random_state)
        labels = model.labels_

        X_sub, lab_sub = subsample(X, labels, n=sample_size, random_state=random_state)
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            emb = reducer.fit_transform(X_sub)
            method = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2, random_state=random_state).fit_transform(X_sub)
            method = "PCA (UMAP unavailable)"

        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=lab_sub, cmap="tab10",
                             s=4, alpha=0.5)
        ax.set_title(f"{method} — {label}", fontsize=11)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")

    plt.tight_layout()
    plt.show()

