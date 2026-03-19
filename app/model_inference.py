import pandas as pd
import mlflow.sklearn
import mlflow
import os

MODEL_URI = os.getenv("MODEL_URI")

model = mlflow.sklearn.load_model(MODEL_URI)


def process_predict(data):

    df = pd.DataFrame(data)

    prediction = model.predict_proba(df)[:, 1].tolist()

    return prediction