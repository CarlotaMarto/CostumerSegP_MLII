"""
utils_clustering.py

Reusable functions for the CLUSTERING stage of the Customer Segmentation
project. The notebook (02_clustering.ipynb) stays thin: it imports these
helpers and the professor's `utils.plot_dendrogram`, and only orchestrates /
narrates. All real logic lives here so it is versionable and DRY.

Design choices baked in here (justified in the notebook / report):
  * The clustering DISTANCE is driven by absolute lifetime spending features
    (`lifetime_spend_*`) plus selected behavioural variables. This preserves
    customer value and category intensity.
  * Groceries are kept in the dataframe for profiling, but alternative feature
    sets exclude `lifetime_spend_groceries` from the distance because it can
    dominate the solution and make several segments look too similar.
  * Features are scaled before KMeans so no single category dominates the
    Euclidean distance purely because of its unit or magnitude.
  * The scaler is fitted ONCE on the regular customers and re-used to project
    the held-aside outliers, so both live in the same feature space.

Functions mirror the sklearn calls used by the professor (KMeans + inertia
elbow, AgglomerativeClustering + plot_dendrogram, confusion_matrix).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix


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

def fit_scaler(df, feature_cols, scaler=None):
    """Fit a scaler on `feature_cols` and return (X_scaled, fitted_scaler).

    StandardScaler by default, matching the Week-6 walkthrough. Fitting and
    transforming are kept separate so the SAME fitted scaler can later project
    the outlier set into the identical feature space.
    """
    scaler = scaler or StandardScaler()
    X = scaler.fit_transform(df[feature_cols].astype(float))
    return X, scaler


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


def silhouette_scan(X, k_range=range(2, 11), random_state=0,
                    n_init=10, sample_size=8000):
    """Return a DataFrame of mean silhouette per k (higher = better separated)."""
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state,
                        n_init=n_init).fit_predict(X)
        s = silhouette_score(X, labels,
                             sample_size=min(sample_size, len(labels)),
                             random_state=random_state)
        rows.append({"k": k, "silhouette": round(float(s), 4)})
    return pd.DataFrame(rows)


def plot_silhouette(sil_df):
    """Plot the silhouette-vs-k curve."""
    plt.figure(figsize=(9, 5))
    plt.plot(sil_df["k"], sil_df["silhouette"], marker="o", color="#1B4F72")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Mean silhouette")
    plt.title("Silhouette score by k")
    plt.xticks(sil_df["k"])
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
    """Fit Agglomerative clustering on a random SAMPLE.

    Agglomerative clustering is O(n^2) in memory, so on tens of thousands of
    customers it cannot run on the full data. We fit it on a representative
    sample purely to CROSS-CHECK the K-Means structure (dendrogram shape and a
    confusion matrix), exactly as the Week-6 notebook compares ward vs k-means.

    Returns
    -------
    (sample_index, ward_labels)
    """
    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X[idx])
    return idx, labels


def fit_dendrogram_model(X, sample_size=5000, linkage="ward", random_state=0):
    """Fit a full-tree Agglomerative model on a sample, for `plot_dendrogram`.

    distance_threshold=0 / n_clusters=None builds the whole tree so the
    professor's `utils.plot_dendrogram` can draw it.
    """
    rng = np.random.RandomState(random_state)
    n = min(sample_size, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    model = AgglomerativeClustering(
        linkage=linkage, distance_threshold=0, n_clusters=None
    ).fit(X[idx])
    return idx, model


def compare_solutions(labels_a, labels_b, name_a="KMeans", name_b="Ward"):
    """Confusion matrix between two clustering label vectors (same rows)."""
    cm = confusion_matrix(labels_a, labels_b)
    k = cm.shape[0]
    return pd.DataFrame(
        cm,
        index=[f"{name_a} {i}" for i in range(k)],
        columns=[f"{name_b} {j}" for j in range(cm.shape[1])],
    )


# ============================================================
# Profiling and visualisation of the final segments
# ============================================================

def profile_clusters(df, label_col, feature_cols, as_percent=False):
    """Mean of `feature_cols` per cluster, with the overall mean as an anchor.

    Comparing each cluster against the OVERALL mean (not against each other)
    is the interpretation rule stressed in the walkthrough.
    """
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


# ============================================================
# Re-attaching the held-aside outliers and exporting
# ============================================================

def assign_outliers(outlier_df, feature_cols, scaler, kmeans_model):
    """Assign each held-aside outlier to its nearest K-Means centroid.

    Outliers are projected with the SAME fitted scaler and labelled with the
    SAME K-Means model, so every customer ends up in a segment without letting
    the extreme values distort the centroids that were learned on the regular
    customers.
    """
    Xo = transform_with_scaler(outlier_df, feature_cols, scaler)
    return kmeans_model.predict(Xo)


def build_full_assignment(regular_df, regular_labels,
                          outlier_df=None, outlier_labels=None,
                          label_col="cluster", id_name="customer_id"):
    """Combine regular + outlier labels into one customer_id -> cluster table.

    Guarantees every customer appears exactly once (the project requires the
    final CSV to contain all customers).
    """
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
    """Mean silhouette per (scaler, k) for Standard / MinMax / Robust.

    Returns a tidy DataFrame; use `plot_scaler_comparison` to draw it. Lets us
    pick the scaler that gives the best-separated clusters on the chosen
    feature space, instead of hard-coding one.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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


def best_scaler_at_k(scaler_df, k):
    """Return the scaler name with the highest silhouette at a given k."""
    sub = scaler_df[scaler_df["k"] == k]
    return sub.loc[sub["silhouette"].idxmax(), "scaler"]


# ============================================================
# Silhouette "blade" plot (per-sample silhouette by cluster)
# ============================================================

def plot_silhouette_blades(X, labels, sample_size=10000, random_state=0,
                           title="Silhouette plot - KMeans"):
    """Draw the per-cluster silhouette blades with the overall average line.

    On large data we evaluate on a random sample for speed; the shape and the
    average are representative of the full solution.
    """
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
    """Return a dict of candidate feature sets to evaluate for clustering.

    `logabs_*` sets log1p the absolute spend columns to tame their skew.
    Demographic / engagement blocks are added only where they exist.
    """
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
# Exploratory helpers — you choose, these only compute/plot
# ============================================================

def get_scaler(name):
    """Return a fresh scaler instance by name: 'Standard' | 'MinMax' | 'Robust' | 'None'.

    'None' (or the value None) means do NOT scale — useful when the features are
    already on a common, comparable scale and you want their natural variance to
    drive the distance. Returns None in that case (apply_feature_pipeline then
    leaves the matrix unscaled).
    """
    if name is None or str(name).lower() == "none":
        return None
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    table = {"Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}
    if name not in table:
        raise ValueError(f"Unknown scaler '{name}'. Choose from {list(table) + ['None']}.")
    return table[name]()


def apply_feature_pipeline(df, cols, logabs=False, scaler=None, fit=False):
    """Build a scaled matrix for `cols`, optionally log1p-ing absolute spend.

    Use fit=True (with a fresh scaler) on the training rows, then fit=False
    with the SAME scaler to project new rows (e.g. outliers) identically.
    """
    X = df[cols].astype(float).copy()
    if logabs:
        for c in X.columns:
            if c.startswith("lifetime_spend_"):
                X[c] = np.log1p(X[c].clip(lower=0))
    if scaler is None:
        return X.to_numpy()
    return scaler.fit_transform(X) if fit else scaler.transform(X)


def silhouette_grid(df, candidate_sets, k_range=range(2, 11), scaler_name="Standard",
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
    """Heatmap of the silhouette grid (feature sets x k). Read it yourself; nothing is chosen."""
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
