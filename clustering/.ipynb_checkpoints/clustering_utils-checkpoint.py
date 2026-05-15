import warnings
from typing import Iterable, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances_argmin_min
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def read_csv_safely(filepath: str) -> pd.DataFrame:
    """Read a CSV file and return an empty DataFrame if the file is missing."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def read_newdata(filepath: str = "info_clustering.csv") -> pd.DataFrame:
    """Read the cleaned non-outlier clustering dataset."""
    return read_csv_safely(filepath)


def read_outliers(filepath: str = "outlier_dataset.csv") -> pd.DataFrame:
    """Read the extreme-outlier dataset that will be assigned to the final clusters."""
    return read_csv_safely(filepath)


def set_customer_index(df: pd.DataFrame, id_col: str = "customer_id") -> pd.DataFrame:
    """Set customer_id as index when it exists."""
    df = df.copy()
    if id_col in df.columns:
        df = df.set_index(id_col)
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[Sequence[str]] = None) -> list:
    """Return numeric feature columns after excluding ID, descriptive, geographic, and label columns."""
    if exclude_cols is None:
        exclude_cols = []

    label_like_cols = [
        col for col in df.columns
        if col.endswith("_cluster") or col in ["cluster", "macro_cluster", "kmeans_cluster", "kmeans20_cluster"]
    ]

    cols_to_drop = list(dict.fromkeys(list(exclude_cols) + label_like_cols))
    candidate_cols = [col for col in df.columns if col not in cols_to_drop]
    numeric_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols


def prepare_features(df: pd.DataFrame, exclude_cols: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, list]:
    """Return the feature matrix used by clustering and the selected feature names."""
    feature_cols = get_feature_columns(df, exclude_cols)
    X = df[feature_cols].copy()

    if X.isna().sum().sum() > 0:
        warnings.warn("Missing values found in clustering features. Filling missing values with column medians.")
        X = X.fillna(X.median(numeric_only=True))

    return X, feature_cols


def run_kmeans(
    df: pd.DataFrame,
    k: int,
    exclude_cols: Optional[Sequence[str]] = None,
    random_state: int = 42,
    n_init: int = 20
):
    """Run KMeans and return labels, centroids, silhouette score, fitted model, and feature names."""
    X, feature_cols = prepare_features(df, exclude_cols)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)
    centroids = pd.DataFrame(model.cluster_centers_, columns=feature_cols)
    centroids.index.name = "cluster"

    return labels, centroids, score, model, feature_cols


def evaluate_kmeans_range(
    df: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 10,
    exclude_cols: Optional[Sequence[str]] = None,
    random_state: int = 42,
    n_init: int = 20,
    plot: bool = True
) -> pd.DataFrame:
    """Evaluate KMeans for a range of k values using inertia and silhouette score."""
    X, _ = prepare_features(df, exclude_cols)

    results = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, labels)
        })

    results_df = pd.DataFrame(results)

    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(results_df["k"], results_df["inertia"], marker="o")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("Inertia")

        ax2 = ax1.twinx()
        ax2.plot(results_df["k"], results_df["silhouette"], marker="x", linestyle="--")
        ax2.set_ylabel("Silhouette score")

        plt.title("KMeans evaluation: elbow and silhouette")
        fig.tight_layout()
        plt.show()

    return results_df


def get_color_map(labels, cmap_name: str = "tab10") -> list:
    """Return one color per label."""
    labels = np.asarray(labels)
    unique_labels = sorted(np.unique(labels))
    cmap = cm.get_cmap(cmap_name, max(len(unique_labels), 1))
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
    return [color_map[label] for label in labels]


def visualize_embedding(
    embedding: np.ndarray,
    labels=None,
    title: str = "2D projection",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    cmap_name: str = "tab10"
):
    """Plot a 2D embedding such as PCA or UMAP."""
    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    else:
        labels = np.asarray(labels)
        colors = get_color_map(labels, cmap_name)
        for label in sorted(np.unique(labels)):
            idx = labels == label
            color = colors[np.where(idx)[0][0]]
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c=[color], s=10, label=f"Cluster {label}")

        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_pca(df: pd.DataFrame, labels, exclude_cols: Optional[Sequence[str]] = None, title: str = "PCA projection"):
    """Fit PCA with 2 components and plot the result."""
    X, _ = prepare_features(df, exclude_cols)
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X)

    visualize_embedding(
        embedding,
        labels=labels,
        title=f"{title} | explained variance: {pca.explained_variance_ratio_.sum():.2%}",
        xlabel="PC 1",
        ylabel="PC 2"
    )

    return embedding, pca


def visualize_umap(df: pd.DataFrame, labels, exclude_cols: Optional[Sequence[str]] = None, title: str = "UMAP projection"):
    """Fit UMAP with 2 components and plot the result."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Please install umap-learn before using visualize_umap: pip install umap-learn") from exc

    X, _ = prepare_features(df, exclude_cols)
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = umap_model.fit_transform(X)

    visualize_embedding(embedding, labels=labels, title=title)

    return embedding, umap_model


def silhouette_plot(X, cluster_labels, title: str = "Silhouette plot", cmap_name: str = "tab10"):
    """Draw a silhouette plot for a fitted clustering solution."""
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    cluster_labels = np.asarray(cluster_labels)
    silhouette_avg = silhouette_score(X_values, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_values, cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10

    for label in sorted(np.unique(cluster_labels)):
        ith_values = sample_silhouette_values[cluster_labels == label]
        ith_values.sort()

        size_i = ith_values.shape[0]
        y_upper = y_lower + size_i

        color = cm.get_cmap(cmap_name)(int(label) % 10)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(label))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Average = {silhouette_avg:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return silhouette_avg


def cluster_profile(df: pd.DataFrame, cluster_col: str, exclude_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Create a cluster profiling table with the mean of each numeric feature and the cluster size."""
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = get_feature_columns(df, exclude_cols + [cluster_col])
    profile = df.groupby(cluster_col)[feature_cols].mean().round(3)
    profile.insert(0, "cluster_size", df[cluster_col].value_counts().sort_index())
    return profile


def relative_cluster_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Compare each cluster mean with the global mean from the profile table."""
    feature_cols = [col for col in profile.columns if col != "cluster_size"]
    weighted_global_mean = np.average(
        profile[feature_cols],
        weights=profile["cluster_size"],
        axis=0
    )
    relative = profile[feature_cols].div(weighted_global_mean, axis=1).round(2)
    relative.insert(0, "cluster_size", profile["cluster_size"])
    return relative


def hierarchical_on_centroids(
    centroids: pd.DataFrame,
    n_macro_clusters: int = 6,
    method: str = "ward",
    plot: bool = True
):
    """Apply hierarchical clustering to KMeans centroids."""
    Z = linkage(centroids.values, method=method)
    macro_labels = fcluster(Z, t=n_macro_clusters, criterion="maxclust")

    centroid_to_macro = {
        centroid_label: macro_label
        for centroid_label, macro_label in zip(centroids.index, macro_labels)
    }

    if plot:
        plt.figure(figsize=(10, 6))
        dendrogram(
            Z,
            labels=[str(i) for i in centroids.index],
            leaf_rotation=90,
            leaf_font_size=10
        )
        plt.title("Hierarchical clustering on KMeans centroids")
        plt.ylabel("Ward distance")
        plt.tight_layout()
        plt.show()

    return Z, macro_labels, centroid_to_macro


def assign_outliers_to_centroids(
    outliers_df: pd.DataFrame,
    centroids: pd.DataFrame,
    exclude_cols: Optional[Sequence[str]] = None,
    cluster_col: str = "kmeans_cluster"
) -> pd.DataFrame:
    """Assign outlier records to the nearest final KMeans centroid."""
    outliers = outliers_df.copy()
    X_outliers, _ = prepare_features(outliers, exclude_cols)

    missing = [col for col in centroids.columns if col not in X_outliers.columns]
    if missing:
        raise ValueError(f"Outlier dataset is missing centroid feature columns: {missing}")

    X_outliers = X_outliers[centroids.columns]

    closest_idx, closest_dist = pairwise_distances_argmin_min(
        X=X_outliers.values,
        Y=centroids.values,
        metric="euclidean"
    )

    centroid_labels = list(centroids.index)
    outliers[cluster_col] = [centroid_labels[i] for i in closest_idx]
    outliers["distance_to_centroid"] = closest_dist

    return outliers