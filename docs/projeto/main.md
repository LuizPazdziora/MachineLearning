# Conclusões

## Comparação dos modelos supervisionados

| Modelo | Acurácia | Precisão (churn) | Recall (churn) | F1 (churn) |
| --- | --- | --- | --- | --- |
| Árvore de decisão (sem limite) | 0,73 | 0,51 | 0,49 | 0,50 |
| Árvore de decisão (`max_depth=5`) | 0,79 | 0,66 | 0,44 | 0,52 |
| KNN (k = 5) | 0,74 | 0,50 | 0,51 | 0,51 |
| **SVM RBF (`class_weight="balanced"`)** | 0,74 | 0,51 | **0,77** | **0,61** |

> As árvores usam divisão 70/30; KNN e SVM usam 80/20 estratificada. Todos com `random_state=42`.

## O que os resultados mostram

1. **Acurácia engana em bases desbalanceadas.** A árvore rasa tem a maior acurácia, mas identifica menos da metade dos clientes que cancelam.
2. **Para retenção, recall de churn importa mais.** O SVM balanceado encontra 77% dos churners, ao custo de mais falsos positivos.
3. **A segmentação explica o problema.** O K-Means isolou um grupo de clientes novos, com contrato mensal e mensalidade alta, em que quase metade cancela.

## Recomendação de negócio

Usar o SVM (ou um modelo equivalente) como **filtro de risco** e priorizar o segmento de alto risco do K-Means em campanhas de **migração para contrato anual**, ajustando o limiar de decisão ao custo de cada ação de retenção.

## Próximos passos técnicos

- Validação cruzada e busca de hiperparâmetros para todos os modelos.
- Testar regressão logística, Random Forest e gradient boosting (XGBoost/LightGBM).
- `OneHotEncoder` para variáveis nominais em vez de `LabelEncoder`.
- Análise de custo × benefício para escolher o limiar ideal.
