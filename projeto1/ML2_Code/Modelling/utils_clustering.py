import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram
import math


def visualize_dimensionality_reduction(transformation: np.array,
                                       targets: list) -> None:
    '''
    Function to plot a scatter plot of the t-SNE or UMAP output.

    Arguments:
        ----------
         - transformation(np.array): Array of t-SNE/UMAP output
         - targets(list): Series containing the assigned
         cluster of all observations

    Returns:
        ----------
         - None, although a plot is produced.
    '''
    # create a scatter plot of the t-SNE output
    ax = sns.scatterplot(x=transformation[:, 0], y=transformation[:, 1],
                         hue=targets, legend='full', edgecolor=None)
    sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def plot_dendrogram(model, **kwargs) -> None:
    '''
    Create linkage matrix and then plot the dendrogram

    Arguments:
        ----------
         - model(HierarchicalClustering Model): hierarchical clustering model.
         - **kwargs

    Returns:
        ----------
         None, but a dendrogram is produced.
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


def plot_comparing_avr_clusters(clusters_centroids_data: pd.DataFrame,
                                colum_to_keep: str) -> None:
    '''
    Create a plot with the mean distribution per cluster group

    Arguments:
        ----------
         - clusters_centroids_data(pd.DataFrame): dataframe
         grouped by cluster.
         - colum_to_keep(str): name of the column containing
         the label of each cluster

    Returns:
        ----------
         None, but a scatterplot is produced.
    '''
    _mean_row = ['mean'] + list(clusters_centroids_data.iloc[:, 1:].mean())

    cluster_centroids_analysis = clusters_centroids_data._append(pd.Series
                                 (_mean_row,
                                  index=list(clusters_centroids_data.columns)),
                                  ignore_index=True)

    # Normalice the values
    cluster_centroids_analysis.iloc[:, 1:] = list(MinMaxScaler().fit_transform
                                     (cluster_centroids_analysis.iloc[:, 1:]))

    # Transform to long format
    melt_data = pd.melt(cluster_centroids_analysis,
                        id_vars=colum_to_keep,
                        var_name='variable',
                        value_name='value')

    sns.set_style("whitegrid")
    sns.scatterplot(melt_data, x='value', y='variable',
                    hue=colum_to_keep, s=100)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


def boxplot_grid(data: pd.DataFrame, variables: list,
                 color: str = None) -> None:
    """
    Plot a boxploty grid based on the data.

    Parameters:
        ----------
         - data (pd.DataFrame): The DataFrame containing the data.
         - variables (list): The column names of the variables to be plotted.
         - color (str, optional): Color for the bars. Defaults to None.
         - edgecolor (str, optional): Color for the bars edges.
         Defaults to 'black'.

    Returns:
        ----------
         None, but a plot is produced
    """
    a = math.ceil(len(variables)/3)
    b = 3
    fig, axes = plt.subplots(a, b, figsize=(25, 75))
    axes = axes.flatten()
    for i, column in enumerate(variables):
        sns.boxplot(x=data.Final_Cluster, y=column,
                    data=data, ax=axes[i], color=color)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].set_title(column)
    axes_to_turn_off = (a * b) - len(variables)
    for i in range(1, axes_to_turn_off + 1):
        axes[-i].axis('off')
    plt.tight_layout()
    plt.show()
