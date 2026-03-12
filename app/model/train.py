import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


train = pd.read_csv("app/data/train.csv")
validation = pd.read_csv("app/data/validation.csv")


features = [
    "tenure",
    "monthly_usage",
    "support_tickets"
]

target = "churn_next_month"


X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]


# busca simples de hiperparâmetro

best_model = None
best_auc = 0

for depth in [3,5,7]:

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_val)[:,1]

    auc = roc_auc_score(y_val, pred)

    print("Depth:", depth, "Validation AUC:", round(auc,3))

    if auc > best_auc:
        best_auc = auc
        best_model = model


print("Best validation AUC:", round(best_auc,3))


# salvar modelo

with open("app/model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Modelo salvo")