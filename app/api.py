from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from .model_inference import process_predict
from ml.schema.pydantic_schema import FeatureRow
from ml.schema.data_definition import ID_COLUMN


class InputData(BaseModel):
    features: List[FeatureRow]


main = FastAPI()


@main.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([f.model_dump() for f in data.features])

    ids = df[ID_COLUMN]

    df_model = df.drop(columns=[ID_COLUMN], errors="ignore")

    pred = process_predict(df_model)

    result = [
        {
            ID_COLUMN: cid,
            "prediction": float(p)
        }
        for cid, p in zip(ids, pred)
    ]

    return {"predictions": result}