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
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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


# ============================================================
# Embedded feature selection / importance (post-hoc on the segments)
# ============================================================
#
# The correlation matrix used in the EDA notebook is a FILTER method: it flags
# redundant features but cannot say which features actually DRIVE the segments.
# Embedded methods answer that by letting a model's own training do the
# selection. Because clustering is unsupervised, we use the cluster label as a
# pseudo-target and run the embedded model post-hoc: this VALIDATES and explains
# the chosen feature set rather than pre-selecting it.
#
#   * Lasso  : multinomial L1-penalised logistic regression. The L1 penalty
#              shrinks uninformative coefficients to exactly zero, so the
#              surviving non-zero coefficients are the embedded selection.
#   * Forest : a Random Forest reads off impurity-based feature_importances_.
#
# A sparse, peaked importance profile means the segmentation rests on a few
# interpretable drivers; a flat profile warns the feature set may be noisy.

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
# Standardised profile heatmap + combined elbow/silhouette
# ============================================================
#
# plot_profile_heatmap shows means in ORIGINAL units (best for naming personas).
# But as a heatmap it is unreadable when features differ in magnitude (spend in
# thousands vs age in tens): the colour scale is swamped by the largest column.
# This z-scores EACH feature across clusters first, so the colour shows how far
# above (warm) or below (cool) the overall average each segment sits per feature
# — i.e. exactly which features separate the segments. Read it together with the
# original-unit table, not instead of it.

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
# Principled feature-set search (granular combos + embedded ranking)
# ============================================================
#
# "Don't dump everything into the distance." These helpers let you compare
# curated combinations on separation (silhouette) AND on which features the
# segmentation actually relies on (embedded importance). The recommended
# workflow: cluster on a broad set, drop the near-zero-importance features,
# then keep the smallest set whose silhouette is essentially unchanged but
# whose surviving features are interpretable for marketing.

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


def rank_features_for_clustering(df, cols, k, scaler_name="Standard",
                                 logabs=True, method="both", random_state=0):
    """Cluster on `cols`, then return the embedded importance of each feature.

    One call that ties the pieces together: scale -> KMeans(k) -> embedded
    importance. Use it to spot features the segmentation ignores (importance
    near zero) so they can be dropped before re-clustering.
    """
    X = apply_feature_pipeline(df, cols, logabs, get_scaler(scaler_name), fit=True)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
    return embedded_feature_importance(X, labels, cols, method=method,
                                       random_state=random_state)


# ============================================================
# Ensemble / consensus clustering (robustness of the segmentation)
# ============================================================
#
# A single KMeans depends on its random initialisation. An ensemble runs it many
# times, ALIGNS the cluster labels across runs (a permutation problem solved with
# the Hungarian algorithm on the confusion matrix) and majority-votes each
# customer's segment. This yields a consensus labelling plus a per-customer
# stability score = fraction of runs that agree with the consensus. High stability
# means the structure is real and not an artefact of one lucky seed; low-stability
# customers sit between segments and can be flagged in the report.

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
        row, col = linear_sum_assignment(-cm)          # match run labels to ref
        mapping = {c: r for r, c in zip(row, col)}
        aligned.append(np.array([mapping.get(x, x) for x in lab]))

    A = np.vstack(aligned).T                            # (n_points, n_runs)
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
