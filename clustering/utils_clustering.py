# utils_clustering.py
# Utility functions for Customer Segmentation - Clustering Notebook

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================================================================
# 1. Metrics and Model Evaluation
# =========================================================================

def evaluate_clustering(X, labels, model_name):
    '''
    Calculates the Silhouette Score and returns a DataFrame with the model summary.
    
    Arguments:
    - X(array-like): Input data.
    - labels(array-like): Cluster labels.
    - model_name(string): Name of the clustering model.
    
    Returns:
    - DataFrame with model evaluation metrics.
    '''
    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return pd.DataFrame({"Model": [model_name], "Clusters": [n_clusters], "Silhouette Score": [np.nan]})
    return pd.DataFrame({"Model": [model_name], "Clusters": [n_clusters], "Silhouette Score": [silhouette_score(X, labels)]})

def cluster_distribution(labels):
    '''
    Returns the absolute count of records per cluster.
    
    Arguments:
    - labels(array-like): Cluster labels.
    
    Returns:
    - Series with counts per cluster.
    '''
    return pd.Series(labels).value_counts().sort_index()

def create_cluster_profile(df_clustered, cluster_col, id_col="customer_id"):
    '''
    Creates a descriptive profile containing the mean of each variable per cluster,
    along with the absolute size and percentage of each group.
    
    Arguments:
    - df_clustered(pd.DataFrame): DataFrame with cluster assignments.
    - cluster_col(string): Name of the cluster column.
    - id_col(string): Name of the customer ID column (default: "customer_id").
    
    Returns:
    - DataFrame with cluster profile.
    '''
    df_profile = df_clustered.drop(columns=[id_col], errors="ignore")
    profile = df_profile.groupby(cluster_col).mean(numeric_only=True)
    cluster_size = df_clustered[cluster_col].value_counts().sort_index()
    cluster_percentage = np.round(cluster_size / len(df_clustered) * 100, 2)
    profile.insert(0, "cluster_size", cluster_size)
    profile.insert(1, "cluster_percentage", cluster_percentage)
    return profile.round(2)

def get_top_cluster_features(profile, top_n=5):
    '''
    Returns a DataFrame indicating the characteristics with highest and lowest
    means in each cluster, helping interpret the profiles.
    
    Arguments:
    - profile(pd.DataFrame): Cluster profile DataFrame.
    - top_n(int): Number of top features to return (default: 5).
    
    Returns:
    - DataFrame with top positive and negative features per cluster.
    '''
    profile_features = profile.drop(columns=["cluster_size", "cluster_percentage"], errors="ignore")
    rows = []
    for cluster in profile_features.index:
        sorted_values = profile_features.loc[cluster].sort_values(ascending=False)
        rows.append({
            "cluster": cluster, 
            "top_positive_features": list(sorted_values.head(top_n).index), 
            "top_negative_features": list(sorted_values.tail(top_n).index)
        })
    return pd.DataFrame(rows)

def evaluate_multiple_kmeans(X, k_range=range(2, 13), random_state=42):
    '''
    Evaluates KMeans for multiple k values and returns a DataFrame with results.
    
    Arguments:
    - X(array-like): Input data.
    - k_range(range): Range of k values to test.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - DataFrame with k, inertia and silhouette_score for each k.
    '''
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette_score(X, labels)
        })
    return pd.DataFrame(results)

def compare_kmeans_models(X, df_original, k_values=[3, 4, 5, 6, 7], random_state=42):
    '''
    Compares multiple KMeans models and returns a dictionary with results.
    
    Arguments:
    - X(array-like): Input data.
    - df_original(pd.DataFrame): Original DataFrame with customer info.
    - k_values(list): List of k values to test.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - Dictionary with model results for each k.
    '''
    results = {}
    for k in k_values:
        print(f"--- KMEANS k={k} ---")
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        
        temp_df = df_original.copy()
        temp_df[f"kmeans_cluster_{k}"] = labels
        
        profile = create_cluster_profile(temp_df, cluster_col=f"kmeans_cluster_{k}")
        metrics = evaluate_clustering(X, labels, model_name=f"KMeans k={k}")
        
        results[k] = {
            "model": model,
            "labels": labels,
            "df": temp_df,
            "profile": profile,
            "metrics": metrics
        }
        
        print("\nCluster distribution:")
        print(cluster_distribution(labels))
        
        print("\nMetrics:")
        print(metrics)
        
        print("\n" + "-" * 50)
    
    return results

def train_kmeans_final(X, n_clusters=4, random_state=42):
    '''
    Trains the final KMeans model with the specified number of clusters.
    
    Arguments:
    - X(array-like): Input data.
    - n_clusters(int): Number of clusters.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - model: Trained KMeans model.
    - labels: Cluster labels.
    '''
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X)
    return model, labels

# =========================================================================
# 2. Plots and Visualizations
# =========================================================================

def plot_elbow_and_silhouette(kmeans_results):
    '''
    Plots side by side the Elbow Method (Inertia) and Silhouette Score means.
    
    Arguments:
    - kmeans_results(pd.DataFrame): DataFrame with k, inertia and silhouette_score.
    
    Returns:
    - None, but a plot is produced.
    '''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(kmeans_results["k"], kmeans_results["inertia"], marker="o", color="b")
    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True)
    
    # Silhouette plot
    axes[1].plot(kmeans_results["k"], kmeans_results["silhouette_score"], marker="o", color="orange")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(X, labels, title):
    '''
    Applies PCA to reduce data to 2 dimensions and plots the found clusters.
    
    Arguments:
    - X(array-like): Input data.
    - labels(array-like): Cluster labels.
    - title(string): Title of the plot.
    
    Returns:
    - embedding: PCA embedding coordinates.
    - pca: Fitted PCA model.
    '''
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)
    print(f"Explained variance with 2 components: {pca.explained_variance_ratio_.sum():.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette="Set2", s=40)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.show()
    return embedding, pca

def plot_silhouette_analysis(X, labels, title="Silhouette Plot"):
    '''
    Generates the silhouette analysis plot for each individual cluster.
    
    Arguments:
    - X(array-like): Input data.
    - labels(array-like): Cluster labels.
    - title(string): Title of the plot.
    
    Returns:
    - None, but a plot is produced.
    '''
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
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
        y_lower = y_upper + 10
        
    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Average = {silhouette_avg:.3f}")
    plt.title(title)
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster")
    plt.legend()
    plt.show()

def plot_cluster_sizes(df, cluster_col):
    '''
    Plots a bar chart with the distribution of customers by cluster.
    
    Arguments:
    - df(pd.DataFrame): DataFrame with cluster assignments.
    - cluster_col(string): Name of the cluster column.
    
    Returns:
    - None, but a plot is produced.
    '''
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=cluster_col, order=sorted(df[cluster_col].unique()))
    plt.title("Customer Distribution by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.show()

def plot_cluster_feature_bars(df, cluster_col, features):
    '''
    Generates bar charts showing the mean value of specific features by cluster.
    
    Arguments:
    - df(pd.DataFrame): DataFrame with cluster assignments.
    - cluster_col(string): Name of the cluster column.
    - features(list): List of feature names to plot.
    
    Returns:
    - None, but plots are produced.
    '''
    for col in features:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x=cluster_col, y=col, errorbar=None)
            plt.title(f"Average {col} by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel(col)
            plt.show()
        else:
            print(f"Skipped missing feature: {col}")

def plot_centroid_comparison(centroids_df, cluster_col="cluster"):
    '''
    Plots a comparative scatter plot with normalized (MinMax) means
    of variables per cluster, facilitating visual identification of patterns.
    
    Arguments:
    - centroids_df(pd.DataFrame): DataFrame with centroids.
    - cluster_col(string): Name of the cluster column.
    
    Returns:
    - None, but a plot is produced.
    '''
    plot_df = centroids_df.copy()
    if cluster_col not in plot_df.columns:
        plot_df.insert(0, cluster_col, plot_df.index)
    features = plot_df.drop(columns=[cluster_col])
    
    scaled_features = pd.DataFrame(MinMaxScaler().fit_transform(features), columns=features.columns, index=plot_df[cluster_col])
    long_df = scaled_features.reset_index().melt(id_vars=cluster_col, var_name="feature", value_name="scaled_value")
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=long_df, x="scaled_value", y="feature", hue=cluster_col, s=90, palette="tab10")
    plt.title("Scaled Centroid Comparison by Cluster")
    plt.xlabel("Scaled centroid value")
    plt.ylabel("Feature")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_tsne_clusters(X, labels, title="t-SNE Visualization", 
                       sample_size=3000, perplexity=30, random_state=42):
    '''
    Applies t-SNE and plots the clusters.
    
    Arguments:
    - X(array-like): Input data.
    - labels(array-like): Cluster labels.
    - title(string): Title of the plot.
    - sample_size(int): Number of samples to use (if None, uses all).
    - perplexity(int): t-SNE perplexity parameter.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - embedding: t-SNE embedding coordinates.
    '''
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
        if isinstance(labels, pd.Series):
            labels_sample = labels[X_sample.index]
        else:
            labels_sample = labels[X_sample.index]
    else:
        X_sample = X
        labels_sample = labels
    
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                learning_rate="auto", init="pca", random_state=random_state)
    embedding = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], 
                    hue=labels_sample, palette="Set2", s=40)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster")
    plt.show()
    return embedding

def plot_umap_clusters(X, labels, title="UMAP Visualization",
                       sample_size=3000, n_neighbors=15, min_dist=0.1, 
                       random_state=42):
    '''
    Applies UMAP and plots the clusters (requires umap-learn package).
    
    Arguments:
    - X(array-like): Input data.
    - labels(array-like): Cluster labels.
    - title(string): Title of the plot.
    - sample_size(int): Number of samples to use.
    - n_neighbors(int): UMAP n_neighbors parameter.
    - min_dist(float): UMAP min_dist parameter.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - embedding: UMAP embedding coordinates, or None if umap not installed.
    '''
    try:
        import umap.umap_ as umap
        
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=random_state)
            if isinstance(labels, pd.Series):
                labels_sample = labels[X_sample.index]
            else:
                labels_sample = labels[X_sample.index]
        else:
            X_sample = X
            labels_sample = labels
        
        umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                               min_dist=min_dist, random_state=random_state)
        embedding = umap_model.fit_transform(X_sample)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], 
                        hue=labels_sample, palette="Set2", s=40)
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(title="Cluster")
        plt.show()
        return embedding
    except ImportError:
        print("UMAP is not installed. Skipping UMAP visualization.")
        return None

def plot_cluster_agreement_heatmap(df, col1, col2, title="Cluster Agreement"):
    '''
    Plots a heatmap of agreement between two segmentations.
    
    Arguments:
    - df(pd.DataFrame): DataFrame with cluster columns.
    - col1(string): First cluster column name.
    - col2(string): Second cluster column name.
    - title(string): Title of the plot.
    
    Returns:
    - agreement: Agreement matrix.
    '''
    agreement = pd.crosstab(df[col1], df[col2], normalize="index")
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(agreement, annot=True, cmap="Blues", fmt=".2f")
    plt.title(title)
    plt.xlabel(f"Cluster - {col2}")
    plt.ylabel(f"Cluster - {col1}")
    plt.show()
    return agreement

# =========================================================================
# 3. Hierarchical Clustering
# =========================================================================

def plot_dendrogram_sample(X, sample_size=1000, method="ward", 
                           truncate_p=50, cut_line=None, random_state=42):
    '''
    Plots a dendrogram from a sample of the data.
    
    Arguments:
    - X(array-like): Input data.
    - sample_size(int): Number of samples to use.
    - method(string): Linkage method.
    - truncate_p(int): Number of leaves to show.
    - cut_line(float): Optional horizontal cut line.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - linkage_matrix: Linkage matrix for the dendrogram.
    '''
    X_sample = X.sample(sample_size, random_state=random_state)
    linkage_matrix = linkage(X_sample, method=method)
    
    plt.figure(figsize=(14, 8))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=truncate_p, leaf_rotation=90)
    
    if cut_line:
        plt.axhline(y=cut_line, color="red", linestyle="--", label=f"Cut at {cut_line}")
        plt.legend()
    
    plt.title(f"Hierarchical Clustering Dendrogram ({method} linkage)")
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    plt.show()
    
    return linkage_matrix

def run_hierarchical_clustering(X, n_clusters=4, linkage_method="ward"):
    '''
    Runs hierarchical clustering and returns labels and metrics.
    
    Arguments:
    - X(array-like): Input data.
    - n_clusters(int): Number of clusters.
    - linkage_method(string): Linkage method.
    
    Returns:
    - Dictionary with model results.
    '''
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    metrics = evaluate_clustering(X, labels, model_name=f"Hierarchical {linkage_method} k={n_clusters}")
    
    return {
        "model": model,
        "labels": labels,
        "metrics": metrics,
        "cluster_distribution": cluster_distribution(labels)
    }

# =========================================================================
# 4. DBSCAN and Mean Shift
# =========================================================================

def run_dbscan_analysis(X, eps=1.5, min_samples=10, title="DBSCAN"):
    '''
    Runs and evaluates DBSCAN.
    
    Arguments:
    - X(array-like): Input data.
    - eps(float): DBSCAN eps parameter.
    - min_samples(int): DBSCAN min_samples parameter.
    - title(string): Model name for evaluation.
    
    Returns:
    - Dictionary with model results.
    '''
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    
    metrics = evaluate_clustering(X, labels, model_name=title)
    
    return {
        "model": model,
        "labels": labels,
        "metrics": metrics,
        "cluster_distribution": cluster_distribution(labels)
    }

def run_meanshift_analysis(X, sample_size=5000, quantile=0.2, bin_seeding=True, random_state=42):
    '''
    Runs and evaluates Mean Shift (on a sample).
    
    Arguments:
    - X(array-like): Input data.
    - sample_size(int): Number of samples to use.
    - quantile(float): Bandwidth estimation quantile.
    - bin_seeding(bool): Whether to use bin seeding.
    - random_state(int): Random seed for reproducibility.
    
    Returns:
    - Dictionary with model results.
    '''
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} points for Mean Shift")
    else:
        X_sample = X
    
    bandwidth = estimate_bandwidth(X_sample, quantile=quantile, 
                                    n_samples=min(1000, len(X_sample)), 
                                    random_state=random_state)
    print(f"Estimated bandwidth: {bandwidth:.4f}")
    
    model = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    labels = model.fit_predict(X_sample)
    
    metrics = evaluate_clustering(X_sample, labels, model_name="Mean Shift")
    
    return {
        "model": model,
        "labels": labels,
        "metrics": metrics,
        "bandwidth": bandwidth,
        "cluster_distribution": cluster_distribution(labels),
        "sample_used": X_sample
    }

# =========================================================================
# 5. Model Comparison
# =========================================================================

def compare_all_models(X, model_results_dict):
    '''
    Compiles metrics from all models into a comparison DataFrame.
    
    Arguments:
    - X(array-like): Input data (for any additional calculations).
    - model_results_dict(dict): Dictionary with model results.
    
    Returns:
    - DataFrame with comparison metrics.
    '''
    all_metrics = []
    for model_name, results in model_results_dict.items():
        if "metrics" in results:
            all_metrics.append(results["metrics"])
    return pd.concat(all_metrics, ignore_index=True)

def print_model_recommendation(comparison_df, primary_metric="Silhouette Score"):
    '''
    Prints recommendation based on model comparison.
    
    Arguments:
    - comparison_df(pd.DataFrame): DataFrame with model comparisons.
    - primary_metric(string): Metric to use for comparison.
    
    Returns:
    - Name of the recommended model.
    '''
    best_model = comparison_df.loc[comparison_df[primary_metric].idxmax()]
    
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"RECOMMENDATION: {best_model['Model']}")
    print(f"  - Clusters: {best_model['Clusters']}")
    print(f"  - {primary_metric}: {best_model[primary_metric]:.4f}")
    print("=" * 60)
    
    return best_model['Model']

# =========================================================================
# 6. Self-Organizing Map (SOM)
# =========================================================================

def train_som(X, grid_size=(12, 12), n_iterations=2500, 
              learning_rate=0.5, random_state=42, sample_size=None):
    '''
    Trains a Self-Organizing Map using MiniSom.
    
    Arguments:
    - X(array-like): Input data (DataFrame or array).
    - grid_size(tuple): (x, y) dimensions of the SOM grid.
    - n_iterations(int): Number of training iterations.
    - learning_rate(float): Initial learning rate.
    - random_state(int): Random seed for reproducibility.
    - sample_size(int): Optional sample size.
    
    Returns:
    - som_weights: SOM weights.
    - som_bmus: Best Matching Units for each point.
    - som_errors: Quantization errors.
    '''
    try:
        from minisom import MiniSom
    except ImportError:
        raise ImportError("MiniSom is required for SOM. Install with: pip install minisom")
    
    if sample_size and len(X) > sample_size:
        X_used = X.sample(sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} points for SOM training")
    else:
        X_used = X
    
    X_array = X_used.values if isinstance(X_used, pd.DataFrame) else X_used
    
    # Normalize data
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_array)
    
    # Initialize and train SOM
    np.random.seed(random_state)
    som = MiniSom(grid_size[0], grid_size[1], X_normalized.shape[1], 
                  sigma=1.0, learning_rate=learning_rate, random_seed=random_state)
    som.random_weights_init(X_normalized)
    som.train_random(X_normalized, n_iterations)
    
    # Calculate BMUs and errors
    som_bmus = np.array([som.winner(x) for x in X_normalized])
    som_errors = np.array([np.linalg.norm(x - som.get_weights()[som_bmus[i][0], som_bmus[i][1]]) 
                           for i, x in enumerate(X_normalized)])
    
    return som.get_weights(), som_bmus, som_errors

def plot_som_hit_map(som_bmus, grid_size=(12, 12), title="SOM Hit Map - Customer Density"):
    '''
    Plots the hit map (customer density) of the SOM.
    
    Arguments:
    - som_bmus(array): Best Matching Units for each point.
    - grid_size(tuple): (x, y) dimensions of the SOM grid.
    - title(string): Title of the plot.
    
    Returns:
    - hit_map: The hit map matrix.
    '''
    hit_map = np.zeros(grid_size)
    for bmu in som_bmus:
        hit_map[bmu[0], bmu[1]] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(hit_map, cmap='hot_r', interpolation='nearest')
    plt.colorbar(label='Number of customers')
    plt.title(title)
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.show()
    return hit_map

def plot_som_quantization_errors(som_errors):
    '''
    Plots histogram of SOM quantization errors.
    
    Arguments:
    - som_errors(array): Quantization errors.
    
    Returns:
    - None, but a plot is produced.
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(som_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=som_errors.mean(), color='red', linestyle='--', label=f'Mean: {som_errors.mean():.4f}')
    plt.axvline(x=np.median(som_errors), color='green', linestyle='--', label=f'Median: {np.median(som_errors):.4f}')
    plt.title('SOM Quantization Errors Distribution')
    plt.xlabel('Quantization Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_som_feature_maps(som_weights, feature_names, selected_features=None, grid_size=(12, 12)):
    '''
    Plots feature maps of the SOM to visualize patterns.
    
    Arguments:
    - som_weights(array): SOM weights.
    - feature_names(list): List of feature names.
    - selected_features(list): Optional list of features to plot.
    - grid_size(tuple): (x, y) dimensions of the SOM grid.
    
    Returns:
    - None, but plots are produced.
    '''
    if selected_features is None:
        selected_features = feature_names[:9]  # Limit to 9 by default
    
    n_features = len(selected_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(selected_features):
        if feature in feature_names:
            feature_idx = feature_names.index(feature)
            feature_map = som_weights[:, :, feature_idx]
            
            im = axes[idx].imshow(feature_map, cmap='viridis', interpolation='nearest')
            axes[idx].set_title(f'{feature}')
            axes[idx].set_xlabel('SOM X')
            axes[idx].set_ylabel('SOM Y')
            plt.colorbar(im, ax=axes[idx])
        else:
            axes[idx].text(0.5, 0.5, f'{feature}\n(not found)', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{feature}')
    
    # Hide any unused subplots
    for idx in range(len(selected_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('SOM Feature Maps', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# =========================================================================
# 7. Outlier Assignment and Recommendations
# =========================================================================

def assign_outliers_to_nearest_centroid(outlier_df, feature_columns, kmeans_model, cluster_col="kmeans_cluster"):
    '''
    Assigns outliers to the nearest centroid of the trained KMeans model.
    
    Arguments:
    - outlier_df(pd.DataFrame): DataFrame with outlier customers.
    - feature_columns(list): List of feature column names.
    - kmeans_model: Trained KMeans model.
    - cluster_col(string): Name of the cluster column to assign.
    
    Returns:
    - DataFrame with assigned clusters.
    '''
    X_outliers = outlier_df[feature_columns].values
    distances = kmeans_model.transform(X_outliers)
    assigned_clusters = np.argmin(distances, axis=1)
    
    outlier_df_assigned = outlier_df.copy()
    outlier_df_assigned[cluster_col] = assigned_clusters
    
    return outlier_df_assigned

def add_segment_names(df, cluster_col="kmeans_cluster", name_col="segment_name", 
                      segment_mapping=None):
    '''
    Adds descriptive names to clusters based on custom mapping.
    
    Arguments:
    - df(pd.DataFrame): DataFrame with cluster assignments.
    - cluster_col(string): Name of the cluster column.
    - name_col(string): Name for the segment name column.
    - segment_mapping(dict): Dictionary mapping cluster numbers to segment names.
    
    Returns:
    - DataFrame with segment names.
    - segment_mapping: The mapping used.
    '''
    if segment_mapping is None:
        # Default mapping - adjust according to your segments
        segment_mapping = {
            0: "Standard Customers",
            1: "Premium Customers", 
            2: "Occasional Customers",
            3: "Loyal Customers",
            4: "High Spenders"
        }
    
    df[name_col] = df[cluster_col].map(segment_mapping)
    return df, segment_mapping

def add_recommendations(df, name_col="segment_name", action_col="recommended_action",
                        recommendation_mapping=None):
    '''
    Adds business recommendations based on customer segment.
    
    Arguments:
    - df(pd.DataFrame): DataFrame with segment names.
    - name_col(string): Name of the segment name column.
    - action_col(string): Name for the recommendation column.
    - recommendation_mapping(dict): Dictionary mapping segments to recommendations.
    
    Returns:
    - DataFrame with recommendations.
    '''
    if recommendation_mapping is None:
        recommendation_mapping = {
            "Standard Customers": "Increase engagement with loyalty program",
            "Premium Customers": "Offer exclusive premium products",
            "Occasional Customers": "Send reactivation campaigns with discounts",
            "Loyal Customers": "Reward loyalty with VIP benefits",
            "High Spenders": "Personalized high-value product recommendations"
        }
    
    df[action_col] = df[name_col].map(recommendation_mapping)
    return df

# =========================================================================
# 8. Data Export
# =========================================================================

def export_segmentation_results(df_clustered, profile_df, comparison_df, 
                                output_dir="../datasets", prefix="customer_segments"):
    '''
    Exports all segmentation results to CSV files.
    
    Arguments:
    - df_clustered(pd.DataFrame): DataFrame with cluster assignments.
    - profile_df(pd.DataFrame): Cluster profile DataFrame.
    - comparison_df(pd.DataFrame): Model comparison DataFrame.
    - output_dir(string): Output directory path.
    - prefix(string): Prefix for output filenames.
    
    Returns:
    - Dictionary with output file paths.
    '''
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure customer_id is stored as integer
    if "customer_id" in df_clustered.columns:
        df_clustered["customer_id"] = df_clustered["customer_id"].astype(int)
    
    # Export main datasets
    final_path = os.path.join(output_dir, f"{prefix}_final.csv")
    df_clustered.to_csv(final_path, index=False)
    
    # Export main columns only if they exist
    main_cols = ["customer_id", "kmeans_cluster", "segment_name", "recommended_action"]
    if all(col in df_clustered.columns for col in main_cols):
        main_path = os.path.join(output_dir, f"{prefix}_main_only.csv")
        df_clustered[main_cols].to_csv(main_path, index=False)
    else:
        main_path = None
    
    # Export profile and comparison
    if profile_df is not None:
        profile_path = os.path.join(output_dir, "kmeans_cluster_profile.csv")
        profile_df.to_csv(profile_path)
    else:
        profile_path = None
    
    if comparison_df is not None:
        comparison_path = os.path.join(output_dir, "clustering_model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
    else:
        comparison_path = None
    
    print(f"All files exported successfully to {output_dir}/")
    
    return {
        "final_dataset": final_path,
        "main_only": main_path,
        "profile": profile_path,
        "comparison": comparison_path
    }

# =========================================================================
# 9. Data Loading and Validation
# =========================================================================

def load_clustering_data(filepath, selected_features, id_col="customer_id"):
    '''
    Loads clustering-ready data and returns X and df.
    
    Arguments:
    - filepath(string): Path to the CSV file.
    - selected_features(list): List of feature names to use.
    - id_col(string): Name of the customer ID column.
    
    Returns:
    - X: Feature matrix.
    - df: Original DataFrame with all columns.
    '''
    df = pd.read_csv(filepath)
    X = df[selected_features]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features used: {len(selected_features)}")
    
    return X, df

def validate_clustering_input(X, min_samples=2):
    '''
    Validates input data for clustering.
    
    Arguments:
    - X(array-like): Input data.
    - min_samples(int): Minimum number of samples required.
    
    Returns:
    - True if validation passes.
    '''
    if X.isnull().any().any():
        raise ValueError("Missing values found in input data. Please handle before clustering.")
    
    if len(X) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples for meaningful clustering.")
    
    print("Input validation passed successfully.")
    return True