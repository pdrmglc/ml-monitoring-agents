from datetime import datetime, timezone
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from .model_inference import process_predict, save_prediction
from ml.schema.pydantic_schema import FeatureRow
from ml.schema.data_definition import ID_COLUMN


class InputData(BaseModel):
    features: List[FeatureRow]


main = FastAPI()


@main.post("/predict")
def predict(data: InputData, save: bool = True):

    df = pd.DataFrame([f.model_dump() for f in data.features])

    ids = df[ID_COLUMN]

    df_model = df.drop(columns=[ID_COLUMN], errors="ignore")

    pred = process_predict(df_model)

    df_output = df_model.copy()
    df_output[ID_COLUMN] = ids.values
    df_output["prediction"] = pred
    df_output["timestamp"] = datetime.now(timezone.utc)

    if save:
        save_prediction(df_output)

    result = [
        {
            ID_COLUMN: cid,
            "prediction": float(p)
        }
        for cid, p in zip(ids, pred)
    ]

    return {"predictions": result}