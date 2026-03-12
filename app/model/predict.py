import pickle
import pandas as pd


# carregar modelo uma única vez
with open("app/model/model.pkl", "rb") as f:
    model = pickle.load(f)


FEATURES = [
    "tenure",
    "monthly_usage",
    "support_tickets"
]


def predict_churn(features: dict) -> float:
    """
    Recebe um dicionário de features e retorna probabilidade de churn.
    """

    X = pd.DataFrame([features])[FEATURES]

    prob = model.predict_proba(X)[0, 1]

    return float(prob)