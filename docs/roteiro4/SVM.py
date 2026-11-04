import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

# ============================================================
# 1. Carregar a base de churn preprocessada
# ============================================================

url = "data/Customer-Churn-preprocessed.csv"
df = pd.read_csv(url)

target_col = "Churn"
if target_col not in df.columns:
    raise ValueError(f"Coluna alvo '{target_col}' não encontrada na base.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Garantia básica: tudo numérico
if not np.all([np.issubdtype(dt, np.number) for dt in X.dtypes]):
    raise TypeError("Existem colunas não numéricas em X. Verifique o pré-processamento.")

# ============================================================
# 2. Padronizar as features e reduzir para 2D com PCA
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA em 2 componentes para visualização (plano 2D)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Variância explicada pelos 2 primeiros componentes:", pca.explained_variance_ratio_)

# ============================================================
# 3. Plotar fronteiras de decisão para diferentes kernels
# ============================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

kernels = {
    "linear": ax1,
    "sigmoid": ax2,
    "poly": ax3,
    "rbf": ax4
}

for k, ax in kernels.items():
    # Modelo SVM com kernel k
    svm = SVC(kernel=k, C=1, gamma="scale")
    svm.fit(X_pca, y)

    # Fronteira de decisão no espaço dos 2 PCs
    DecisionBoundaryDisplay.from_estimator(
        svm,
        X_pca,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        ax=ax
    )

    # Pontos de dados (clientes)
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        s=20,
        edgecolors="k"
    )

    ax.set_title(k)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

fig.suptitle("Fronteiras de decisão – SVM em 2 PCs (Customer Churn)", y=1.02)
plt.tight_layout()

# ============================================================
# 4. Exportar a figura (execute a partir da raiz do repositório)
# ============================================================

plt.savefig("docs/roteiro4/SVMplt.png", dpi=100, bbox_inches="tight")

plt.close()