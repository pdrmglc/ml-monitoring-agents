import pandas as pd
from ml.schema.data_definition import NUMERICAL_COLUMNS, ALL_COLUMNS


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # garante colunas esperadas
    missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas faltando: {missing_cols}")

    # força tipo numérico
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[ALL_COLUMNS]