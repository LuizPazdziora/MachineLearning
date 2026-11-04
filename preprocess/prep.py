import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------
# Configurações da base Telco Customer Churn
# -------------------------------------------------------------------
CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn",
]

NUM_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

ID_COLS = ["customerID"]
TARGET_COL = "Churn"


# -------------------------------------------------------------------
# 1) Carregamento do dataset original
# -------------------------------------------------------------------
def load_churn_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset Telco Customer Churn a partir de um arquivo CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


# -------------------------------------------------------------------
# 2) Limpeza básica (sem padronização)
# -------------------------------------------------------------------
def clean_churn_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica os passos de limpeza básicos:
    - remove colunas de ID (se existirem)
    - trata TotalCharges (string com espaços → numérico)
    - converte colunas numéricas
    - preenche NAs de categóricas com a moda
    - preenche NAs de numéricas com a mediana
    """
    df = df.copy()

    # Remove ID
    df = df.drop(columns=ID_COLS, errors="ignore")

    # Tratamento especial de TotalCharges (string → numérico)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)
            .str.strip()      # remove espaços
            .replace({"": np.nan})  # string vazia → NaN
        )

    # Converter colunas numéricas
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preencher valores faltantes em categóricas com a moda
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Preencher valores faltantes em numéricas com a mediana
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# -------------------------------------------------------------------
# 3) Codificação categórica (LabelEncoder), sem StandardScaler
# -------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica LabelEncoder em cada coluna categórica definida em CAT_COLS,
    incluindo a coluna alvo 'Churn'.

    Retorna um DataFrame já numérico (ou quase todo numérico),
    pronto para ser salvo como CSV preprocessado.
    """
    df = df.copy()
    label_encoder = LabelEncoder()

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    return df


# -------------------------------------------------------------------
# 4) Pipeline de preparação (sem padronização)
# -------------------------------------------------------------------
def prepare_churn_base(csv_in: str | Path) -> pd.DataFrame:
    """
    Pipeline de preparação da base churn, SEM standardization:

    1) Carrega o CSV original
    2) Limpa a base (ID, NAs, tipos, TotalCharges, etc.)
    3) Codifica colunas categóricas com LabelEncoder

    Retorna
    -------
    df_prepared : pd.DataFrame
        DataFrame limpo e codificado, pronto para uso em modelagem.
    """
    df_raw = load_churn_data(csv_in)
    df_clean = clean_churn_df(df_raw)
    df_prepared = encode_categorical(df_clean)

    return df_prepared


# -------------------------------------------------------------------
# 5) Execução direta: gerar e salvar CSV preprocessado
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Caminho de entrada: base original
    ROOT = Path(__file__).resolve().parents[1]
    INPUT_CSV = ROOT / "data" / "Customer-Churn.csv"

    # Caminho de saída: base já limpa e codificada
    OUTPUT_CSV = ROOT / "data" / "Customer-Churn-preprocessed.csv"

    df_preprocessed = prepare_churn_base(INPUT_CSV)

    # Exporta CSV sem índice
    df_preprocessed.to_csv(OUTPUT_CSV, index=False)
    print(f"Base preprocessada salva em: {OUTPUT_CSV}")
