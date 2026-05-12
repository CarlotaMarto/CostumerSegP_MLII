import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import math

def read_newdata(filepath="info_clustering.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()

def train_som_with_qe(som, data_arr, n_iter, sample_every=100):
    """MiniSom update loop that records quantisation error."""
    qe = []
    for i in range(n_iter):
        idx    = np.random.randint(data_arr.shape[0])
        sample = data_arr[idx]

        som.update(sample, som.winner(sample), i, n_iter)

        if (i + 1) % sample_every == 0:
            qe.append(som.quantization_error(data_arr))
    return qe



def plot_hits_map(som, data, cmap="Blues"):
    """
    Visualise how many samples fall on each SOM neuron.
    """
    data_arr = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)

    grid_x, grid_y = som.get_weights().shape[:2]

    hits = np.zeros((grid_x, grid_y), dtype=int)
    for v in data_arr:
        i, j = som.winner(v)
        hits[i, j] += 1

    plt.figure(figsize=(6, 6))
    im = plt.imshow(hits.T, origin="lower", cmap=cmap)
    plt.colorbar(im, label="Samples per neuron")
    plt.title("Hits map")
    plt.xticks([]); plt.yticks([])

    for i in range(grid_x):
        for j in range(grid_y):
            if hits[i, j] > 0:
                plt.text(i, j, hits[i, j],
                         ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.show()

def evaluate_som(
    data_arr,
    grid_size,
    sigma,
    learning_rate,
    neighborhood_function='gaussian',
    random_seed=0,
    n_iterations=5000,
    sample_every=None
):
    """
    1) Creates a MiniSom with the given hyperparameters
    2) Initializes weights to data distribution
    3) Trains for `n_iterations` iterations (random updates)
    4) Returns (quantization_error, topographic_error)
       computed on the final map.
    """
    input_len = data_arr.shape[1]
    som = MiniSom(
        x=grid_size,
        y=grid_size,
        input_len=input_len,
        sigma=sigma,
        learning_rate=learning_rate,
        neighborhood_function=neighborhood_function,
        random_seed=random_seed
    )
    som.random_weights_init(data_arr)

    if sample_every is None:
        som.train_random(data_arr, n_iterations)
    else:
        som.train_random(data_arr, n_iterations)

    qe = som.quantization_error(data_arr)
    te = som.topographic_error(data_arr)

    return qe, te
    
def plot_component_planes(som, feature_names):
    n_feat = len(feature_names)
    n_cols = 4
    n_rows = int(np.ceil(n_feat / n_cols))

    plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows))
    for k, name in enumerate(feature_names):
        plane = som.get_weights()[:, :, k]   
        plt.subplot(n_rows, n_cols, k + 1)
        plt.imshow(plane.T, origin='lower', cmap='coolwarm')
        plt.title(name); plt.xticks([]); plt.yticks([])
        plt.colorbar(shrink=0.75)
    plt.suptitle('Component planes', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_quant_errors(qe, every):
    it = np.arange(every, every * len(qe) + 1, every)
    plt.figure(figsize=(8, 4))
    plt.plot(it, qe, linewidth=1.5)
    plt.xlabel('Update #')
    plt.ylabel('Quantisation error')
    plt.title('SOM convergence')
    plt.grid(True, ls='--', alpha=.5)
    plt.tight_layout()
    plt.show()

def plot_u_matrix(som):
    umatrix = som.distance_map()
    plt.figure(figsize=(6, 6))
    plt.imshow(umatrix.T, origin='lower', cmap='viridis')
    plt.colorbar(label='Average distance')
    plt.title('U-matrix')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_feature_maps(som, feature_names, grid_size, n_cols=5):
    """
    Plot each feature’s 2D weight‐plane. 
    som.get_weights() has shape (grid_size, grid_size, n_features).
    """
    n_features = len(feature_names)
    n_rows = math.ceil(n_features / n_cols)
    weights = som.get_weights()  

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    for i, fname in enumerate(feature_names):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.set_title(fname, fontsize=10)
        pcm = ax.pcolor(weights[:, :, i].T, cmap='coolwarm')
        plt.colorbar(pcm, ax=ax, fraction=0.045, pad=0.04)

        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(grid_size + 1))
        ax.set_yticks(np.arange(grid_size + 1))
        ax.set_aspect('equal')

    for empty_spot in range(n_features + 1, n_rows * n_cols + 1):
        plt.subplot(n_rows, n_cols, empty_spot).axis('off')

    plt.tight_layout()
    plt.show()