import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# -------------------------------------------------------------------
# 1) Configuração de caminhos (execute a partir da raiz do repositório)
# -------------------------------------------------------------------
DATA_PATH = Path("data/Customer-Churn-preprocessed.csv")

OUTPUT_DIR = Path("docs/roteiro2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # cria pasta se não existir

# Nomes dos arquivos
CM_PATH = OUTPUT_DIR / "cm-knn-churn.svg"


# -------------------------------------------------------------------
# 2) Carregar base
# -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------------------
# 3) Separar X e y
# -------------------------------------------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

# -------------------------------------------------------------------
# 4) StandardScaler
# -------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------------------------------------------
# 5) Split treino/teste
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------------------
# 6) Treinar KNN
# -------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# -------------------------------------------------------------------
# 7) MATRIZ DE CONFUSÃO (EXPORTAR IMAGEM)
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - KNN (Churn)")

plt.savefig(CM_PATH, format="svg", transparent=True, bbox_inches="tight")
plt.close()
