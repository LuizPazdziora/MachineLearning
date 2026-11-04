import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay
)

 
# 1. Carregar base
 

url = "data/Customer-Churn-preprocessed.csv"
df = pd.read_csv(url)

target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

 
# 2. Padronização
 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 
# 3. Separar treino/teste
 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

 
# 4. Treinar SVM (kernel RBF)
 

svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train, y_train)

 
# 5. Predições
 

y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

 
# 6. Métricas
 

print("Matriz de confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {accuracy:.4f}")
print(f"ROC-AUC : {roc_auc:.4f}")

 
# 7. Curva ROC
 

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC – SVM (Churn)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("docs/roteiro4/Curva-ROC-SVM.png", dpi=100)
plt.close()
