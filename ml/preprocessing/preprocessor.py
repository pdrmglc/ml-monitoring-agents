import pandas as pd

from ml.encoders.oof_target_encoder import OOFMeanTargetEncoder
from ml.schema.data_definition import (
    NUMERICAL_COLUMNS,
    BIN_COLUMNS,
    CATEGORICAL_COLUMNS,
)

# mapeamento padrão para binários
BINARY_MAP = {
    "Yes": 1,
    "No": 0,
    "No internet service": 0,
    "No phone service": 0,
}


class Preprocessor:

    def __init__(self):
        self.categorical_features = CATEGORICAL_COLUMNS
        self.encoder = OOFMeanTargetEncoder(self.categorical_features)
        self.binary_map = BINARY_MAP

    def _normalize_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Assumo que BIN_COLUMNS só tem colunas que são ou numéricas (0/1) ou categóricas com valores mapeáveis para binário
        for col in BIN_COLUMNS:

            # se já for numérico, só garante tipo
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            else:
                df[col] = df[col].map(self.binary_map)

            # valida valores finais (tem que ser 0 ou 1)
            if not df[col].dropna().isin([0, 1]).all():
                raise ValueError(f"Valor inválido encontrado na coluna binária: {col}")

        return df

    def _enforce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in NUMERICAL_COLUMNS:

            # trata string vazia
            df[col] = df[col].replace("", None)

            # converte
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # valida apenas valores não nulos
            invalid_mask = df[col].notna() & ~pd.to_numeric(df[col], errors="coerce").notna()

            if invalid_mask.any():
                raise ValueError(f"Valores inválidos na coluna numérica: {col}")

        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = X.copy()

        # 1. normaliza binários
        X = self._normalize_binary_columns(X)

        # 2. garante numéricos
        X = self._enforce_numeric(X)

        # 3. encoding (somente categóricos reais)
        X_encoded = self.encoder.fit_transform(X, y)

        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. normaliza binários
        X = self._normalize_binary_columns(X)

        # 2. garante numéricos
        X = self._enforce_numeric(X)

        # 3. encoding
        X_encoded = self.encoder.transform(X)

        return X_encoded