import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Carregar base real
df = pd.read_csv(
    "data/Customer-Churn-preprocessed.csv"
)


# Remover Churn
X = df.drop(columns=["Churn"])


# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA (2 dimensões)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


# K-Means
k = 3
kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=100, random_state=42)
labels = kmeans.fit_predict(X_pca)


# Plot
plt.figure(figsize=(12, 10))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=40,
    alpha=0.7
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="*",
    s=250,
    label="Centróides"
)

plt.title("K-Means aplicado à base Telco Customer Churn (PCA 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

# Exportar a figura (execute a partir da raiz do repositório)
plt.savefig("docs/roteiro3/kmeans_telco_churn.png", dpi=100, bbox_inches="tight")

plt.close()