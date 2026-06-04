"""Utilities for the customer segmentation clustering workflow.

This module centralises feature construction, scaling, clustering diagnostics,
robustness checks and segment profiling used in the clustering notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram


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

def kmeans_elbow(X, k_range=range(1, 11), random_state=0, n_init=10):
    """Compute KMeans inertia across k (the elbow / dispersion curve)."""
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
        inertia.append(km.inertia_)
    return list(k_range), inertia


def plot_elbow(k_values, inertia, cutoffs=None):
    """Plot the inertia elbow, optionally marking candidate cut points."""
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, inertia, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Dispersion (inertia)")
    plt.title("Elbow Method - K-Means")
    plt.xticks(list(k_values))
    if cutoffs:
        for c in cutoffs:
            plt.axvline(c, color="red", linestyle="--", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# Fitting the final solutions
# ============================================================

def fit_kmeans(X, k, random_state=0, n_init=10):
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
    sns.lineplot(data=results_df, x="k", y="silhouette", hue="method", marker="o")
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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
    sns.lineplot(data=r2_df, x="k", y="r2", hue="linkage", marker="o")
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
    sns.barplot(x=sizes.index.astype(str), y=sizes.values, color="#1B4F72")
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.title("Customers per cluster")
    plt.show()


def plot_profile_heatmap(profile_df, title="Cluster profile"):
    """Heatmap of a profile table (clusters x features). Drop OVERALL row first."""
    data = profile_df.drop(index="OVERALL", errors="ignore")
    plt.figure(figsize=(max(8, data.shape[1] * 0.9), max(4, data.shape[0] * 0.7)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5,
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
                      color="#7FB3D5"):
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
        plt.plot(g["k"], g["silhouette"], marker="o", label=name)
    if mark_k is not None:
        plt.axvline(mark_k, color="red", linestyle="--", alpha=0.6,
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
        color = cm.tab10(ci % 10)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                          facecolor=color, edgecolor=color, alpha=0.8)
        plt.text(-0.02, y_lower + size / 2, str(ci))
        y_lower = y_upper + 10
    plt.axvline(avg, color="red", linestyle="--", label=f"Avg = {avg:.2f}")
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


def plot_embedding(embedding, labels, title="Embedding", method_name=None):
    """Scatter a 2-D embedding coloured by cluster label."""
    labels = np.asarray(labels)
    plt.figure(figsize=(9, 7))
    for ci in sorted(np.unique(labels)):
        m = labels == ci
        plt.scatter(embedding[m, 0], embedding[m, 1], s=6, alpha=0.5,
                    label=f"Cluster {ci}")
    plt.title(title if not method_name else f"{title} ({method_name})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.show()


# ============================================================
# Feature-set comparison (which columns to cluster on)
# ============================================================

def build_candidate_feature_sets(df):
    """Return candidate feature spaces to evaluate for clustering."""
    abss = [c for c in df.columns if c.startswith("lifetime_spend_")]
    abss_no_groceries = [c for c in abss if c != "lifetime_spend_groceries"]
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
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", linewidths=0.5,
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
                   color=["#1B4F72", "#7FB3D5"][:len(cols)])
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
    sns.heatmap(z, annot=True, fmt="+.1f", cmap="RdBu_r", center=0,
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
    plt.plot(curve_df["iteration"], curve_df["quantization_error"], marker="o")
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
        cmap="viridis",
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
        cmap="Blues",
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
                          cmap="RdYlBu_r"):
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
    plt.hist(stability, bins=20, color="#1B4F72", edgecolor="white")
    plt.axvline(stability.mean(), color="red", linestyle="--",
                label=f"mean = {stability.mean():.3f}")
    plt.xlabel("Fraction of runs agreeing with the consensus label")
    plt.ylabel("Customers")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
