# Árvore de Decisão

## Objetivo

A base **Telco Customer Churn** reúne dados de 7.043 clientes de uma operadora de telecom. O objetivo é **prever se o cliente vai cancelar (churn) ou permanecer**.

## Dicionário de dados

| Coluna | Descrição |
| --- | --- |
| `customerID` | ID do cliente |
| `gender` | Gênero do cliente (feminino, masculino) |
| `SeniorCitizen` | Se o cliente é idoso ou não (1, 0) |
| `Partner` | Se o cliente tem cônjuge/parceiro (Sim, Não) |
| `Dependents` | Se o cliente possui dependentes (Sim, Não) |
| `tenure` | Número de meses que o cliente permaneceu na empresa |
| `PhoneService` | Se o cliente possui serviço de telefonia (Sim, Não) |
| `MultipleLines` | Se o cliente possui múltiplas linhas (Sim, Não, Sem serviço de telefonia) |
| `InternetService` | Provedor de internet do cliente (DSL, Fibra óptica, Sem internet) |
| `OnlineSecurity` | Se o cliente possui segurança online (Sim, Não, Sem serviço de internet) |
| `OnlineBackup` | Se o cliente possui backup online (Sim, Não, Sem serviço de internet) |
| `DeviceProtection` | Se o cliente possui proteção de dispositivos (Sim, Não, Sem serviço de internet) |
| `TechSupport` | Se o cliente possui suporte técnico (Sim, Não, Sem serviço de internet) |
| `StreamingTV` | Se o cliente possui streaming de TV (Sim, Não, Sem serviço de internet) |
| `StreamingMovies` | Se o cliente possui streaming de filmes (Sim, Não, Sem serviço de internet) |
| `Contract` | Tipo de contrato do cliente (Mês a mês, Um ano, Dois anos) |
| `PaperlessBilling` | Se o cliente utiliza fatura digital (Sim, Não) |
| `PaymentMethod` | Método de pagamento do cliente (Cheque eletrônico, Cheque enviado, Transferência bancária — automática, Cartão de crédito — automático) |
| `MonthlyCharges` | Valor cobrado mensalmente do cliente |
| `TotalCharges` | Valor total cobrado do cliente |
| `Churn` | Se o cliente cancelou (Sim ou Não) |

## Pré-processamento

O pré-processamento fica centralizado em [`preprocess/prep.py`](https://github.com/LuizPazdziora/MachineLearning/blob/main/preprocess/prep.py) e é reutilizado por todos os roteiros:

1. **Remoção do identificador** `customerID`.
2. **Padronização de tipos:** `TotalCharges` chega como texto (com espaços em branco para clientes novos) e é convertido para numérico.
3. **Valores faltantes:** numéricos (`tenure`, `MonthlyCharges`, `TotalCharges`) recebem a **mediana**; categóricos recebem a **moda**.
4. **Codificação:** variáveis categóricas (incluindo `Churn`) são convertidas com `LabelEncoder`.

Amostra de 10 clientes após o pré-processamento:

``` python exec="on" html="0"
import sys
sys.path.insert(0, "preprocess")

import pandas as pd
from prep import clean_churn_df, encode_categorical

df_raw = pd.read_csv("data/Customer-Churn.csv")
df = encode_categorical(clean_churn_df(df_raw))

print(df.sample(n=10, random_state=42).to_markdown(index=False))
```

## Divisão dos dados e treinamento

Os dados foram divididos em **70% treino** e **30% validação** (`random_state=42`). Foram treinadas duas árvores: uma **sem limite de profundidade** e outra com **`max_depth=5`**, para comparar o efeito do *overfitting*.

``` python exec="on" html="on"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Customer-Churn-preprocessed.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

resultados = []
for profundidade in [None, 5]:
    modelo = tree.DecisionTreeClassifier(max_depth=profundidade, random_state=42)
    modelo.fit(x_train, y_train)
    pred = modelo.predict(x_test)
    rel = classification_report(y_test, pred, output_dict=True)
    resultados.append({
        "max_depth": "sem limite" if profundidade is None else profundidade,
        "acurácia": round(accuracy_score(y_test, pred), 3),
        "recall churn": round(rel["1"]["recall"], 3),
        "F1 churn": round(rel["1"]["f1-score"], 3),
    })

print(pd.DataFrame(resultados).to_html(index=False))

importancias = (
    pd.DataFrame({"Variável": X.columns, "Importância": modelo.feature_importances_})
    .sort_values("Importância", ascending=False)
    .head(8)
)
print("<h4>Variáveis mais importantes (max_depth=5)</h4>")
print(importancias.to_html(index=False, float_format="%.3f"))

plt.figure(figsize=(14, 6))
tree.plot_tree(modelo, max_depth=2, feature_names=list(X.columns), class_names=["Não churn", "Churn"], filled=True, fontsize=8)
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches="tight")
plt.close()
print(buffer.getvalue())
```

## Avaliação do modelo

- A árvore **sem limite de profundidade** decora o conjunto de treino e chega a cerca de **73%** de acurácia na validação.
- Limitando a profundidade em **5**, a acurácia sobe para cerca de **79%**: o modelo generaliza melhor.
- Mesmo assim, o *recall* da classe churn fica abaixo de 50%, ou seja, a árvore ainda deixa passar muitos clientes que cancelam. Tipo de contrato, segurança online e tempo de casa (`tenure`) são as variáveis mais relevantes.
- Próximos passos: balanceamento de classes, ajuste de hiperparâmetros com validação cruzada e comparação com os demais modelos (ver [Conclusões](../projeto/main.md)).
