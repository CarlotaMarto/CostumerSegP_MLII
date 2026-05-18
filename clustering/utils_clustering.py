"""
Utility functions for the Customer Segmentation clustering notebook.

The functions in this file keep the notebook cleaner and make the clustering
workflow easier to reuse and explain.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min


def evaluate_clustering(X, labels, model_name):
    """
    Evaluates a clustering solution using Silhouette Score.
    If the model produces fewer than two valid clusters, the score is returned as NaN.
    """
    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return pd.DataFrame({
            "Model": [model_name],
            "Clusters": [n_clusters],
            "Silhouette Score": [np.nan]
        })

    return pd.DataFrame({
        "Model": [model_name],
        "Clusters": [n_clusters],
        "Silhouette Score": [silhouette_score(X, labels)]
    })


def cluster_distribution(labels):
    """
    Returns the number of observations assigned to each cluster.
    """
    return pd.Series(labels).value_counts().sort_index()


def create_cluster_profile(df_clustered, cluster_col, id_col="customer_id"):
    """
    Creates a cluster profile with average feature values by cluster.
    Identifier columns are excluded from the profile.
    """
    df_profile = df_clustered.drop(columns=[id_col], errors="ignore")

    profile = df_profile.groupby(cluster_col).mean(numeric_only=True)

    cluster_size = df_clustered[cluster_col].value_counts().sort_index()
    cluster_percentage = np.round(cluster_size / len(df_clustered) * 100, 2)

    profile.insert(0, "cluster_size", cluster_size)
    profile.insert(1, "cluster_percentage", cluster_percentage)

    return profile.round(2)


def plot_elbow_and_silhouette(kmeans_results):
    """
    Plots the Elbow Method and Silhouette Scores for K-Means.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(kmeans_results["k"], kmeans_results["inertia"], marker="o")
    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True)

    axes[1].plot(kmeans_results["k"], kmeans_results["silhouette_score"], marker="o")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_pca_clusters(X, labels, title):
    """
    Visualizes clustering results using the first two PCA components.
    """
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)

    print(f"Explained variance with 2 components: {pca.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels,
        palette="Set2",
        s=40
    )

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.show()

    return embedding, pca


def plot_silhouette_analysis(X, labels, title="Silhouette Plot"):
    """
    Creates a silhouette plot for the selected clustering solution.
    """
    labels = np.array(labels)
    unique_clusters = sorted([cluster for cluster in set(labels) if cluster != -1])

    if len(unique_clusters) < 2:
        print("Silhouette plot requires at least 2 valid clusters.")
        return

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    plt.figure(figsize=(10, 6))

    y_lower = 10
    for cluster in unique_clusters:
        cluster_values = sample_silhouette_values[labels == cluster]
        cluster_values.sort()

        size_cluster = cluster_values.shape[0]
        y_upper = y_lower + size_cluster

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            alpha=0.7
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
        y_lower = y_upper + 10

    plt.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        label=f"Average = {silhouette_avg:.3f}"
    )

    plt.title(title)
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster")
    plt.legend()
    plt.show()


def plot_cluster_sizes(df, cluster_col):
    """
    Plots the number of customers in each cluster.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=cluster_col,
        order=sorted(df[cluster_col].unique())
    )
    plt.title("Customer Distribution by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.show()


def plot_cluster_feature_bars(df, cluster_col, features):
    """
    Plots average feature values by cluster for selected variables.
    """
    for col in features:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.barplot(
                data=df,
                x=cluster_col,
                y=col,
                errorbar=None
            )
            plt.title(f"Average {col} by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel(col)
            plt.show()
        else:
            print(f"Skipped missing feature: {col}")


def get_top_cluster_features(profile, top_n=5):
    """
    Shows the highest and lowest standardized characteristics per cluster.
    """
    profile_features = profile.drop(
        columns=["cluster_size", "cluster_percentage"],
        errors="ignore"
    )

    rows = []

    for cluster in profile_features.index:
        sorted_values = profile_features.loc[cluster].sort_values(ascending=False)

        rows.append({
            "cluster": cluster,
            "top_positive_features": list(sorted_values.head(top_n).index),
            "top_negative_features": list(sorted_values.tail(top_n).index)
        })

    return pd.DataFrame(rows)


def plot_centroid_comparison(centroids_df, cluster_col="cluster"):
    """
    Plots a compact comparison of cluster centroids using MinMax scaling.
    This is only used for visualization.
    """
    plot_df = centroids_df.copy()

    if cluster_col not in plot_df.columns:
        plot_df.insert(0, cluster_col, plot_df.index)

    features = plot_df.drop(columns=[cluster_col])

    scaled_features = pd.DataFrame(
        MinMaxScaler().fit_transform(features),
        columns=features.columns,
        index=plot_df[cluster_col]
    )

    long_df = scaled_features.reset_index().melt(
        id_vars=cluster_col,
        var_name="feature",
        value_name="scaled_value"
    )

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=long_df,
        x="scaled_value",
        y="feature",
        hue=cluster_col,
        s=90,
        palette="tab10"
    )
    plt.title("Scaled Centroid Comparison by Cluster")
    plt.xlabel("Scaled centroid value")
    plt.ylabel("Feature")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def add_segment_names(df, cluster_col="kmeans_cluster", name_col="segment_name"):
    """
    Adds final business segment names based on the interpreted K-Means cluster profile.
    """
    segment_names = {
        0: "Promotion-Oriented Regulars",
        1: "Healthy Lifestyle Buyers",
        2: "Premium Grocery & Pet Buyers",
        3: "Tech-Focused Customers",
        4: "Family Loyal Customers"
    }

    df_named = df.copy()
    df_named[name_col] = df_named[cluster_col].map(segment_names)

    return df_named, segment_names


def add_recommendations(df, name_col="segment_name", action_col="recommended_action"):
    """
    Adds simple business recommendations for each interpreted segment.
    """
    recommendations = {
        "Promotion-Oriented Regulars": "Use discount campaigns, personalized coupons, and promotion bundles.",
        "Healthy Lifestyle Buyers": "Promote fresh produce, hygiene products, wellness bundles, and healthy lifestyle offers.",
        "Premium Grocery & Pet Buyers": "Use cross-category bundles involving meat, fish, pet food, and premium grocery items.",
        "Tech-Focused Customers": "Target with technology offers, new product launches, and digital engagement campaigns.",
        "Family Loyal Customers": "Offer family packs, loyalty rewards, bulk discounts, and household-oriented promotions."
    }

    df_out = df.copy()
    df_out[action_col] = df_out[name_col].map(recommendations)
    return df_out


def assign_outliers_to_nearest_centroid(outlier_df, feature_columns, kmeans_model, cluster_col="kmeans_cluster"):
    """
    Assigns already-preprocessed outliers to the nearest K-Means centroid.

    Important: the outlier dataframe must be in the same feature space and scaling
    as the dataset used to train K-Means.
    """
    missing_cols = [col for col in feature_columns if col not in outlier_df.columns]

    if missing_cols:
        raise ValueError(
            "Outlier dataset is missing required clustering columns: "
            + ", ".join(missing_cols)
        )

    X_outliers = outlier_df[feature_columns]
    labels, distances = pairwise_distances_argmin_min(
        X_outliers,
        kmeans_model.cluster_centers_
    )

    assigned = outlier_df.copy()
    assigned[cluster_col] = labels
    assigned["distance_to_centroid"] = distances

    return assigned

def plot_selected_centroid_heatmap(kmeans_centroids, features, cluster_col="cluster"):
    """
    Plots a heatmap of selected KMeans centroid features.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    selected_features = [
        col for col in features
        if col in kmeans_centroids.columns
    ]

    centroid_plot = kmeans_centroids[
        [cluster_col] + selected_features
    ].set_index(cluster_col)

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        centroid_plot,
        annot=True,
        cmap="RdYlBu_r",
        center=0,
        fmt=".2f"
    )

    plt.title("Selected Centroid Features by Cluster")
    plt.xlabel("Feature")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def train_som(
    X,
    grid_size=(12, 12),
    n_iterations=2500,
    learning_rate=0.5,
    random_state=42,
    sample_size=8000
):
    """
    Trains a simple Self-Organizing Map using NumPy.
    Returns SOM weights, best matching units and quantization errors.
    """

    import numpy as np

    np.random.seed(random_state)

    X_values = X.values if hasattr(X, "values") else X

    if len(X_values) > sample_size:
        sample_idx = np.random.choice(
            len(X_values),
            sample_size,
            replace=False
        )
        X_train = X_values[sample_idx]
    else:
        X_train = X_values.copy()

    rows, cols = grid_size
    n_features = X_train.shape[1]

    weights = np.random.normal(
        size=(rows, cols, n_features)
    )

    weights = weights / np.linalg.norm(
        weights,
        axis=2,
        keepdims=True
    )

    initial_radius = max(rows, cols) / 2

    for iteration in range(n_iterations):
        x = X_train[np.random.randint(0, len(X_train))]

        distances = np.linalg.norm(
            weights - x,
            axis=2
        )

        bmu = np.unravel_index(
            np.argmin(distances),
            distances.shape
        )

        lr = learning_rate * np.exp(
            -iteration / n_iterations
        )

        radius = initial_radius * np.exp(
            -iteration / n_iterations
        )

        for i in range(rows):
            for j in range(cols):
                dist_to_bmu = np.linalg.norm(
                    np.array([i, j]) - np.array(bmu)
                )

                if dist_to_bmu <= radius:
                    influence = np.exp(
                        -(dist_to_bmu ** 2) / (2 * (radius ** 2))
                    )

                    weights[i, j] += lr * influence * (
                        x - weights[i, j]
                    )

    bmus = []
    errors = []

    for x in X_values:
        distances = np.linalg.norm(
            weights - x,
            axis=2
        )

        bmu = np.unravel_index(
            np.argmin(distances),
            distances.shape
        )

        bmus.append(bmu)
        errors.append(distances[bmu])

    return weights, bmus, np.array(errors)


def plot_som_hit_map(som_bmus, grid_size=(12, 12), title="SOM Hit Map"):
    """
    Plots SOM hit map.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    hit_map = np.zeros(grid_size)

    for bmu in som_bmus:
        hit_map[bmu] += 1

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        hit_map,
        cmap="Blues"
    )

    plt.title(title)
    plt.xlabel("SOM Column")
    plt.ylabel("SOM Row")
    plt.show()

def plot_som_quantization_errors(som_errors):
    """
    Plots distribution of SOM quantization errors.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))

    sns.histplot(
        som_errors,
        bins=40,
        kde=True
    )

    plt.title("SOM Quantization Error Distribution")
    plt.xlabel("Quantization Error")
    plt.ylabel("Frequency")
    plt.show()



def plot_som_feature_maps(
    som_weights,
    feature_names,
    selected_features=None,
    grid_size=(12, 12),
    max_features=12
):
    """
    Plots SOM feature maps for selected variables.
    """

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_names = list(feature_names)

    if selected_features is None:
        selected_features = feature_names[:max_features]

    selected_features = [
        feature for feature in selected_features
        if feature in feature_names
    ]

    if len(selected_features) == 0:
        print("No valid features selected.")
        return

    n_cols = 3
    n_rows = math.ceil(len(selected_features) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 5 * n_rows)
    )

    axes = np.array(axes).flatten()

    for idx, feature in enumerate(selected_features):
        feature_idx = feature_names.index(feature)

        sns.heatmap(
            som_weights[:, :, feature_idx],
            ax=axes[idx],
            cmap="RdYlBu_r",
            center=0,
            cbar=True
        )

        axes[idx].set_title(feature)
        axes[idx].set_xlabel("SOM Column")
        axes[idx].set_ylabel("SOM Row")

    for idx in range(len(selected_features), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()