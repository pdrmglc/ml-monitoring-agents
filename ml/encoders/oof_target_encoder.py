import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import KFold


class OOFMeanTargetEncoder:

    def __init__(self, categorical_features: List[str], n_splits: int = 5):
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.global_mean = None
        self.mappings: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):

        self.global_mean = y.mean()

        for col in self.categorical_features:

            mapping = (
                pd.concat([X[col], y], axis=1)
                .groupby(col)[y.name]
                .mean()
                .to_dict()
            )

            self.mappings[col] = mapping

        return self

    def transform(self, X: pd.DataFrame):

        X_transformed = X.copy()

        for col in self.categorical_features:

            mapping = self.mappings[col]

            X_transformed[col] = (
                X_transformed[col]
                .map(mapping)
                .fillna(self.global_mean)
            )

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):

        X_transformed = X.copy()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        self.global_mean = y.mean()

        for col in self.categorical_features:

            oof_values = np.zeros(len(X))

            for train_idx, val_idx in kf.split(X):

                X_tr = X.iloc[train_idx]
                y_tr = y.iloc[train_idx]

                X_val = X.iloc[val_idx]

                mapping = (
                    pd.concat([X_tr[col], y_tr], axis=1)
                    .groupby(col)[y_tr.name]
                    .mean()
                )

                encoded = X_val[col].map(mapping)

                oof_values[val_idx] = encoded.fillna(self.global_mean)

            X_transformed[col] = oof_values

        # depois cria o mapping final para inferência futura
        self.fit(X, y)

        return X_transformed