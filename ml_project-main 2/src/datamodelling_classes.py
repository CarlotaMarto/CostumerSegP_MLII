import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.base import clone
from utils import (
    plot_dendrogram,
    plot_2d_with_classes,
    visualize_dimensionality_reduction
)


class Clusteringworkflow:
    """
    Class containing the workflow of the algorithms used for the creation of clusters.
    """
    def __init__(self, algorithm):
        #instatiation of an object, which is just an algorithm for clustering.
        self.algorithm = algorithm

        
    def find_optimal_k(self, data_scaled, K_range= 25):
        """
        This function, for the algorithm selected, fits it experimentally on the data, and plots either an elbow method for KMeans or a dendrogam
        for AgglomerativeClustering so that we can extract a possible optimal number of clusters (K).

        Parameters:
        data_scaled -> dataframe: The data on which we want to experimental fit the algorithm.
        K_range -> int: The values of K we are going to test the algorithm with. Initially set to 25 because only KMeans uses that parameter, and by setting
        it initially to 25, we don't need to manually pass a number to it, which means when the algorithm is not KMeans, it will just ignore that parameter.

        -----------
        Output:
        A graph, either the inertia plot for KMeans, or a dendogram for AgglomerativeClustering, suggesting an optimal number of clusters.
        """
        if isinstance(self.algorithm, KMeans):
            dispersion = []

            for k in range(2, K_range+1):
                model_kmeans = clone(self.algorithm)
                model_kmeans.set_params(n_clusters=k)
                model_kmeans.fit(data_scaled)
                dispersion.append(model_kmeans.inertia_)

            plt.plot(range(2, K_range+1), dispersion, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Dispersion (inertia)')
            plt.show()

        elif isinstance(self.algorithm, AgglomerativeClustering):
            model_hierarchical = clone(self.algorithm)
            model_hierarchical.set_params(compute_distances=True)
            model_hierarchical.fit(data_scaled)
            fig, ax = plt.subplots()
            plt.title("Hierarchical Clustering Dendrogram")
            # plot the top three levels of the dendrogram
            plot_dendrogram(model_hierarchical, truncate_mode="level", p=50)
            plt.show() 

        else:
            raise ValueError("This is only for KMeans or AgglomerativeClustering")
        
    def silhouette(self, data_scaled, K_range = 25):
        """
        Function with similar objective as the previous one. It also helps to find an optimal number of clusters
        
        Parameters:
        data_scaled -> dataframe: The data on which we want to experimental fit the algorithm.
        K_range -> int: The values of K we are going to test the algorithm with. Initially set to 25 because only KMeans uses that parameter, and by setting
        it initially to 25, we don't need to manually pass a number to it, which means when the algorithm is not KMeans, it will just ignore that parameter.

        --------------
        Output:
        A graph, showing the silhouette score for each value of k
        """
        if not isinstance(self.algorithm, (KMeans, AgglomerativeClustering)):
                raise ValueError("Silhouette only works for KMeans or AgglomerativeClustering")
        
        scores = []

        for k in range(2,K_range+1):
            model = clone(self.algorithm)

            if isinstance(model, AgglomerativeClustering):
                model.set_params(
                    n_clusters=k,
                    distance_threshold=None,
                    compute_full_tree=True
                )
            else:
                model.set_params(n_clusters=k)

            labels = model.fit_predict(data_scaled)
            scores.append(silhouette_score(data_scaled, labels))
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(2,K_range+1), scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores')
        plt.show()
        
    def silhouette_plot(self, data_scaled, labels=None, title="Silhouette Plot"):
        """
        Creates a silhouette diagram for the fitted clustering solution.

        Parameters:
        data_scaled -> array/dataframe: scaled data used for clustering.
        labels -> array-like: cluster labels. If None, labels are generated using self.algorithm.
        title -> str: plot title.

        Output:
        Silhouette plot and mean silhouette score.
        """
        if labels is None:
            labels = self.algorithm.fit_predict(data_scaled)

        labels = np.asarray(labels)

        sample_scores = silhouette_samples(data_scaled, labels)
        overall_score = silhouette_score(data_scaled, labels)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        plt.figure(figsize=(8, 6))

        y_lower = 10

        for cluster_label, color in zip(unique_labels, colors):
            cluster_scores = sample_scores[labels == cluster_label]
            cluster_scores.sort()

            size_cluster = cluster_scores.shape[0]
            y_upper = y_lower + size_cluster

            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_scores,
                facecolor=color,
                edgecolor=color,
                alpha=0.8
            )

            plt.text(
                -0.05,
                y_lower + 0.5 * size_cluster,
                f"Cluster {cluster_label}"
            )

            y_lower = y_upper + 10

        plt.axvline(
            x=overall_score,
            color="red",
            linestyle="--",
            label=f"Mean silhouette = {overall_score:.3f}"
        )

        plt.xlabel("Silhouette coefficient")
        plt.ylabel("Cluster")
        plt.title(title)
        plt.xlim([-0.2, 1])
        plt.legend()
        plt.show()

        return overall_score
    

    def fit_and_prediction(self, original_data, data_scaled):
        """
        This function fits the chosen algorithm to the data previously scaled and creates a column in the original dataset containing the cluster that
        each observation is allocated to.

        Parameters:
        original_data -> dataframe: The dataframe where we'll add the column with the labels for each observation.
        data_scaled -> dataframe: The data on which we'll train the algorithm and create the clusters.
        """
        col_name = f"cluster_{self.algorithm.__class__.__name__.lower()}"
        original_data[col_name] = self.algorithm.fit_predict(data_scaled) 
    
        return self



class DimensionalityReduction:
    """
    This is the class for algorithms that are going to be used with the intent of facilitating visualization
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit_and_transformation(self, data_scaled):
        self.fitted_model = self.algorithm.fit_transform(data_scaled)

        return self
    

class PCAWorkflow(DimensionalityReduction):
    def __init__(self, algorithm):
        if not isinstance(algorithm, PCA):
            raise ValueError("PCAWorkflow only accepts PCA algorithm.")

        super().__init__(algorithm)
    
    def fit_pca(self, data_scaled):
        self.algorithm.fit(data_scaled)

        return self

    def explained_variance(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.algorithm.explained_variance_ratio_.cumsum(), marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('Explained Variance by Components')
        plt.show()

    def plot(self, labels):
        plot_2d_with_classes(labels, self.fitted_model)


class TSNEWorkflow(DimensionalityReduction):
    def __init__(self, algorithm):
        if not isinstance(algorithm, TSNE):
            raise ValueError("TSNEWorkflow only accepts TSNE algorithm.")

        super().__init__(algorithm)
    
    def plot(self, labels):
        visualize_dimensionality_reduction(self.fitted_model, labels)


class UMAPWorkflow(DimensionalityReduction):
    def __init__(self, algorithm):
        if not isinstance(algorithm, UMAP):
            raise ValueError("UMAPWorkflow only accepts UMAP algorithm.")

        super().__init__(algorithm)
    
    def plot(self, labels):
        visualize_dimensionality_reduction(self.fitted_model, labels)    
