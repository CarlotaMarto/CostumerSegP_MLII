import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_local_data(filename="info_clustering_ready.csv"):
    """Carrega os dados processados a partir do diretório de trabalho atual."""
    path = os.path.join(os.getcwd(), filename)
    try:
        df = pd.read_csv(path)
        print(f"Dados carregados: {filename} | Formato: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Erro: O ficheiro {filename} nao foi encontrado no diretorio atual.")
        return pd.DataFrame()

def run_kmeans_pipeline(df, exclude_cols, k):
    """Executa o algoritmo K-Means omitindo colunas de identificacao ou categorias brutas."""
    features = df.drop(columns=exclude_cols, errors='ignore')
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_
    
    # Calculo da silhueta amostrado para otimizar o tempo de execucao
    sample_size = min(5000, len(features))
    score = silhouette_score(features, labels, sample_size=sample_size, random_state=42)
    
    return labels, centroids, score

def plot_elbow_and_silhouette(df, exclude_cols, max_k=10):
    """Gera os graficos do Metodo do Cotovelo e da Silhueta lado a lado."""
    features = df.drop(columns=exclude_cols, errors='ignore')
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)
    
    print("A calcular os indicadores para a escolha de K. Por favor, aguarde.")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        
        sample_size = min(3000, len(features))
        silhouettes.append(silhouette_score(features, labels, sample_size=sample_size, random_state=42))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Grafico do Cotovelo
    ax1.plot(k_range, inertias, 'bo-', color='#0047AB')
    ax1.set_title('Metodo do Cotovelo (Inertia)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Numero de Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    
    # Grafico da Silhueta
    ax2.plot(k_range, silhouettes, 'ro-', color='#D9534F')
    ax2.set_title('Analise de Silhueta', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Numero de Clusters (K)')
    ax2.set_ylabel('Score Medio de Silhueta')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()