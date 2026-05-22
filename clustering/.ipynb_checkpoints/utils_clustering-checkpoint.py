# utils_clustering.py
# Utility functions for Customer Segmentation - Clustering Notebook

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

# =========================================================================
# 1. Metrics and Model Evaluation
# =========================================================================

def evaluate_clustering(X, labels, model_name):
    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return pd.DataFrame({"Model": [model_name], "Clusters": [n_clusters], "Silhouette Score": [np.nan]})
    return pd.DataFrame({"Model": [model_name], "Clusters": [n_clusters], "Silhouette Score": [silhouette_score(X, labels)]})

def cluster_distribution(labels):
    return pd.Series(labels).value_counts().sort_index()

def create_cluster_profile(df_clustered, cluster_col, id_col="customer_id"):
    df_profile = df_clustered.drop(columns=[id_col], errors="ignore")
    profile = df_profile.groupby(cluster_col).mean(numeric_only=True)
    cluster_size = df_clustered[cluster_col].value_counts().sort_index()
    cluster_percentage = np.round(cluster_size / len(df_clustered) * 100, 2)
    profile.insert(0, "cluster_size", cluster_size)
    profile.insert(1, "cluster_percentage", cluster_percentage)
    return profile.round(2)

def get_top_cluster_features(profile, top_n=5):
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
        
        print(f"Cluster distribution: {cluster_distribution(labels).to_dict()}")
        print(f"Silhouette Score: {metrics['Silhouette Score'].values[0]:.4f}")
        print("-" * 50)
    
    return results

def train_kmeans_final(X, n_clusters=4, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X)
    return model, labels

# =========================================================================
# 2. Plots and Visualizations
# =========================================================================

def plot_elbow_and_silhouette(kmeans_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(kmeans_results["k"], kmeans_results["inertia"], marker="o", color="b")
    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True)
    
    axes[1].plot(kmeans_results["k"], kmeans_results["silhouette_score"], marker="o", color="orange")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(X, labels, title):
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

def plot_tsne_clusters(X, labels, title="t-SNE Visualization", sample_size=3000, perplexity=30, random_state=42):
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
        labels_sample = labels[X_sample.index]
    else:
        X_sample = X
        labels_sample = labels
    
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=random_state)
    embedding = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels_sample, palette="Set2", s=40)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster")
    plt.show()
    return embedding

def plot_umap_clusters(X, labels, title="UMAP Visualization", sample_size=3000, n_neighbors=15, min_dist=0.1, random_state=42):
    import umap.umap_ as umap
    
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
        labels_sample = labels[X_sample.index]
    else:
        X_sample = X
        labels_sample = labels
    
    umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = umap_model.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels_sample, palette="Set2", s=40)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="Cluster")
    plt.show()
    return embedding

def plot_kmeans_pca_comparison(kmeans_comparison, X, k_values=[3, 4, 5, 6, 7]):
    fig, axes = plt.subplots(2, len(k_values), figsize=(20, 10))
    fig.suptitle('K-Means Models Comparison - PCA Visualizations', fontsize=16, y=1.02)
    
    for idx, k in enumerate(k_values):
        labels = kmeans_comparison[k]["labels"]
        
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(X)
        
        ax = axes[0, idx]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Set2', s=20, alpha=0.6)
        ax.set_title(f'K-Means k={k}')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        ax = axes[1, idx]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Set2', s=20, alpha=0.4)
        
        centroids = kmeans_comparison[k]["model"].cluster_centers_
        centroids_pca = pca.transform(centroids)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
        ax.set_title(f'K-Means k={k} (with Centroids)')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_kmeans_umap_comparison(kmeans_comparison, X, k_values=[3, 4, 5, 6, 7], sample_size=3000, random_state=42):
    import umap.umap_ as umap
    
    X_sample = X.sample(sample_size, random_state=random_state)
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(25, 5))
    fig.suptitle('K-Means Models Comparison - UMAP Visualizations', fontsize=16, y=1.02)
    
    for idx, k in enumerate(k_values):
        labels_full = kmeans_comparison[k]["labels"]
        labels_sample = labels_full[X_sample.index]
        
        umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=random_state)
        umap_embedding = umap_model.fit_transform(X_sample)
        
        ax = axes[idx]
        scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels_sample, cmap='Set2', s=30, alpha=0.7)
        ax.set_title(f'K-Means k={k}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.tight_layout()
    plt.show()

# =========================================================================
# 3. Hierarchical Clustering
# =========================================================================

def plot_dendrogram_sample(X, sample_size=1000, method="ward", truncate_p=50, cut_line=None, random_state=42):
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
    if sample_size and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
        print(f"Using sample of {sample_size} points for Mean Shift")
    else:
        X_sample = X
    
    bandwidth = estimate_bandwidth(X_sample, quantile=quantile, n_samples=min(1000, len(X_sample)), random_state=random_state)
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

def compare_all_models(model_results_dict):
    all_metrics = []
    for model_name, results in model_results_dict.items():
        if "metrics" in results:
            all_metrics.append(results["metrics"])
    return pd.concat(all_metrics, ignore_index=True)

def print_model_recommendation(comparison_df, primary_metric="Silhouette Score"):
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
# 6. Outlier Assignment and Recommendations
# =========================================================================

def assign_outliers_to_nearest_centroid(outlier_df, feature_columns, kmeans_model, cluster_col="kmeans_cluster"):
    X_outliers = outlier_df[feature_columns].values
    distances = kmeans_model.transform(X_outliers)
    assigned_clusters = np.argmin(distances, axis=1)
    
    outlier_df_assigned = outlier_df.copy()
    outlier_df_assigned[cluster_col] = assigned_clusters
    
    return outlier_df_assigned

def add_segment_names(df, cluster_col="kmeans_cluster", name_col="segment_name", segment_mapping=None):
    if segment_mapping is None:
        segment_mapping = {
            0: "Standard Customers",
            1: "Premium Customers", 
            2: "Occasional Customers",
            3: "Loyal Customers",
            4: "High Spenders"
        }
    
    df[name_col] = df[cluster_col].map(segment_mapping)
    return df, segment_mapping

def add_recommendations(df, name_col="segment_name", action_col="recommended_action", recommendation_mapping=None):
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
# 7. Data Export
# =========================================================================

def export_segmentation_results(df_clustered, profile_df, comparison_df, output_dir="../datasets", prefix="customer_segments"):
    os.makedirs(output_dir, exist_ok=True)
    
    if "customer_id" in df_clustered.columns:
        df_clustered["customer_id"] = df_clustered["customer_id"].astype(int)
    
    final_path = os.path.join(output_dir, f"{prefix}_final.csv")
    df_clustered.to_csv(final_path, index=False)
    
    main_cols = ["customer_id", "kmeans_cluster", "segment_name", "recommended_action"]
    if all(col in df_clustered.columns for col in main_cols):
        main_path = os.path.join(output_dir, f"{prefix}_main_only.csv")
        df_clustered[main_cols].to_csv(main_path, index=False)
    
    if profile_df is not None:
        profile_df.to_csv(os.path.join(output_dir, "kmeans_cluster_profile.csv"))
    
    if comparison_df is not None:
        comparison_df.to_csv(os.path.join(output_dir, "clustering_model_comparison.csv"), index=False)
    
    print(f"All files exported successfully to {output_dir}/")
    
    return {
        "final_dataset": final_path,
        "profile": os.path.join(output_dir, "kmeans_cluster_profile.csv"),
        "comparison": os.path.join(output_dir, "clustering_model_comparison.csv")
    }

# =========================================================================
# 8. Data Loading
# =========================================================================

def load_clustering_data(filepath, selected_features):
    df = pd.read_csv(filepath)
    X = df[selected_features]
    print(f"Dataset shape: {X.shape}")
    print(f"Features used: {len(selected_features)}")
    return X, df