import pandas as pd
import numpy as np

np.random.seed(42)

n_customers = 200
months = 12

rows = []

for month in range(1, months + 1):

    for customer_id in range(n_customers):

        tenure = np.random.randint(1, 36)

        # drift temporal gradual
        usage_mean = 50 + month * 0.8
        monthly_usage = np.random.normal(usage_mean, 8)

        support_tickets = np.random.poisson(1.5 + month * 0.05)

        logit = (
            -0.03 * tenure
            + 0.05 * support_tickets
            - 0.02 * monthly_usage
        )

        prob = 1 / (1 + np.exp(-logit))

        churn_next_month = np.random.binomial(1, prob)

        rows.append({
            "customer_id": customer_id,
            "month": month,
            "tenure": tenure,
            "monthly_usage": monthly_usage,
            "support_tickets": support_tickets,
            "churn_next_month": churn_next_month
        })

df = pd.DataFrame(rows)

df.to_csv("app/data/churn_dataset_full.csv", index=False)

print("Dataset gerado:", df.shape)
print(df.head())