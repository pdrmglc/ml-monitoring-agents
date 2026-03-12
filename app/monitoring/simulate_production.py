import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv("app/data/churn_dataset_full.csv")

model = joblib.load("app/model/model.pkl")

features = ["tenure", "monthly_usage", "support_tickets"]

production_months = sorted(df["month"].unique())
production_months = [m for m in production_months if m > 8]

results = []

for month in production_months:

    month_data = df[df["month"] == month]

    X = month_data[features]
    y_true = month_data["churn_next_month"]

    preds = model.predict(X)

    accuracy = accuracy_score(y_true, preds)

    print(f"month {month} accuracy:", accuracy)

    results.append({
        "month": month,
        "accuracy": accuracy,
        "n_samples": len(month_data)
    })

metrics = pd.DataFrame(results)

metrics.to_csv("app/monitoring/production_metrics.csv", index=False)