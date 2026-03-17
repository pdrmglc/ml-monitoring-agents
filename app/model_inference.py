import pandas as pd
import mlflow.sklearn
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

MODEL_URI = "models:/churn_model@production"

model = mlflow.sklearn.load_model(MODEL_URI)


def process_predict(data):

    df = pd.DataFrame(data)

    prediction = model.predict_proba(df)[:, 1].tolist()

    return prediction