import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, RobustScaler
from matplotlib.lines import Line2D 



def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def scaling(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data



def plot_2d_with_classes(labels, components):
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    custom = [Line2D([], [], marker='.', color=colors[i], linestyle='None') 
              for i in range(len(unique_labels))]
    
    color_code = [colors[np.where(unique_labels == label)[0][0]] for label in labels]
    
    plt.scatter(
        components[:, 0],
        components[:, 1],
        alpha=0.5,
        c=color_code,
        edgecolors='black'
    )
    
    plt.legend(handles=custom,
               labels=[f'Cluster {l}' for l in unique_labels],
               bbox_to_anchor=(1.05, 0.5), loc='lower left')
    
    plt.show()


def visualize_dimensionality_reduction(transformation, targets):
    # create a scatter plot of the t-SNE output
    plt.scatter(transformation[:, 0], transformation[:, 1],
                c=np.array(targets).astype(int), cmap=plt.cm.tab20)

    labels = np.unique(targets)

    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=min(np.array(labels).astype(int)), vmax=max(np.array(labels).astype(int)))
    rgba_values = cmap(norm(labels))

    # create a legend with the class labels and colors
    handles = [plt.scatter([], [], c=rgba, label=label) for rgba, label in zip(rgba_values, labels)]
    plt.legend(handles=handles, title='Classes')
