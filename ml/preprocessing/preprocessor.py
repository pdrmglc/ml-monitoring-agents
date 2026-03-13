from ml.encoders.oof_target_encoder import OOFMeanTargetEncoder


class Preprocessor:

    def __init__(self, categorical_features):

        self.categorical_features = categorical_features
        self.encoder = OOFMeanTargetEncoder(categorical_features)

    def fit_transform(self, X, y):

        X = X.copy()

        X_encoded = self.encoder.fit_transform(X, y)

        return X_encoded

    def transform(self, X):

        X = X.copy()

        X_encoded = self.encoder.transform(X)

        return X_encoded