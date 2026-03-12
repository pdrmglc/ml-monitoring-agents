from app.model.predict import predict_churn
import random

for i in range(10):

    features = {
        "tenure": random.randint(1, 36),
        "monthly_usage": random.uniform(10, 80),
        "support_tickets": random.randint(0, 5)
    }

    prob = predict_churn(features)

    print(f"Prediction {i+1}")
    print("features:", features)
    print("churn_probability:", round(prob, 4))
    print()