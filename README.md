# Machine Learning — Previsão de Churn em Telecom

[![ci](https://github.com/LuizPazdziora/MachineLearning/actions/workflows/main.yaml/badge.svg)](https://github.com/LuizPazdziora/MachineLearning/actions/workflows/main.yaml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![MkDocs](https://img.shields.io/badge/MkDocs_Material-526CFE?logo=materialformkdocs&logoColor=white)

Estudo de classificação e segmentação de clientes na base pública **Telco Customer Churn** (7.043 clientes): quais clientes vão cancelar e quais perfis concentram o risco.

**📄 Relatório completo:** https://luizpazdziora.github.io/MachineLearning/

## Resultados

| Modelo | Acurácia | Recall (churn) | F1 (churn) |
| --- | --- | --- | --- |
| Árvore de decisão (`max_depth=5`) | 0,79 | 0,44 | 0,52 |
| KNN (k = 5) | 0,74 | 0,51 | 0,51 |
| **SVM RBF (balanceado)** | 0,74 | **0,77** | **0,61** |

- O **SVM** foi o modelo que mais detectou clientes que cancelam (ROC-AUC 0,82).
- O **K-Means (k = 3)** isolou um segmento com **~48% de churn**: clientes recentes, com contrato mês a mês e mensalidade alta.

## Estrutura

```text
├── data/                     # base original e pré-processada
├── preprocess/prep.py        # limpeza, imputação e encoding (reutilizado nos roteiros)
├── docs/
│   ├── index.md              # visão geral
│   ├── roteiro1/             # Árvore de decisão
│   ├── roteiro2/             # KNN
│   ├── roteiro3/             # K-Means
│   ├── roteiro4/             # SVM
│   └── projeto/              # Conclusões
└── mkdocs.yml
```

Os blocos de código das páginas são **executados no build** (markdown-exec), então as tabelas e métricas do site são geradas a partir dos dados.

## Como executar

```bash
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt

mkdocs serve -o                # visualizar o site localmente
python preprocess/prep.py      # regenerar a base pré-processada
python docs/roteiro4/SVMmetricas.py   # exemplo: métricas do SVM
```

Os scripts devem ser executados a partir da raiz do repositório. O deploy no GitHub Pages é feito pelo GitHub Actions a cada *push* na `main`.

## Autor

**Luiz Fernando Pazdziora Costa** — Sistemas de Informação e Negócios (ESPM), foco em Ciência de Dados · [GitHub](https://github.com/LuizPazdziora)

Base: *Telco Customer Churn* (IBM Sample Data). Estrutura do site baseada no template de documentação da disciplina.
