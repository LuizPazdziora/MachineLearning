# Machine Learning — Previsão de Churn em Telecom

Estudo de **classificação e segmentação de clientes** na base pública *Telco Customer Churn* (7.043 clientes de uma operadora de telecom). O objetivo é **prever quais clientes vão cancelar** e **identificar perfis de risco** para orientar ações de retenção.

**Autor:** Luiz Fernando Pazdziora Costa — Sistemas de Informação e Negócios (ESPM), foco em Ciência de Dados · [GitHub](https://github.com/LuizPazdziora)

## Etapas

| Etapa | Técnica | Página |
| --- | --- | --- |
| Pré-processamento | Limpeza de tipos, imputação (mediana/moda), `LabelEncoder` | [Árvore de decisão](roteiro1/main.md#pre-processamento) |
| Classificação | Árvore de decisão | [Roteiro 1](roteiro1/main.md) |
| Classificação | KNN (k = 5) | [Roteiro 2](roteiro2/main.md) |
| Segmentação | K-Means (k = 3) + PCA | [Roteiro 3](roteiro3/main.md) |
| Classificação | SVM com kernel RBF | [Roteiro 4](roteiro4/main.md) |
| Síntese | Comparação dos modelos | [Conclusões](projeto/main.md) |

## Resultados em resumo

| Modelo | Acurácia | Recall (churn) | F1 (churn) | Observação |
| --- | --- | --- | --- | --- |
| Árvore de decisão (`max_depth=5`) | 0,79 | 0,44 | 0,52 | Maior acurácia, mas deixa passar mais da metade dos churners |
| KNN (k = 5) | 0,74 | 0,51 | 0,51 | Bom para a classe majoritária |
| **SVM RBF (balanceado)** | 0,74 | **0,77** | **0,61** | Melhor detecção de churn · ROC-AUC 0,82 |

O K-Means encontrou um segmento com **~48% de churn**: clientes recentes, com contrato mês a mês e mensalidade elevada.

## Tecnologias

Python · pandas · scikit-learn · matplotlib · seaborn · MkDocs Material · GitHub Actions (deploy automático no GitHub Pages)
