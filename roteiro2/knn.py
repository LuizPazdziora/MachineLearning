import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# -------------------------------------------------------------------
# 1) Carregar base já preprocessada
# -------------------------------------------------------------------
df = pd.read_csv(
    "data/Customer-Churn-preprocessed.csv"
)

# -------------------------------------------------------------------
# 2) Separar X e y
# -------------------------------------------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

# -------------------------------------------------------------------
# 3) StandardScaler
# -------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------------------
# 4) Split treino/teste
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------------------
# 5) KNN
# -------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

# -------------------------------------------------------------------
# 6) Métrica única: acurácia
# -------------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia do KNN (k=5): {accuracy:.3f}")
