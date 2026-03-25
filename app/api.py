from datetime import datetime, timezone
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from .model_inference import process_predict, save_prediction, get_training_ids
from ml.schema.pydantic_schema import FeatureRow
from ml.schema.data_definition import ID_COLUMN

from app.model_inference import run_drift, run_drift_html
from fastapi.responses import HTMLResponse

class InputData(BaseModel):
    features: List[FeatureRow]


main = FastAPI()


@main.post("/predict")
def predict(data: InputData, save: bool = True):

    df = pd.DataFrame([f.model_dump() for f in data.features])

    ids = df[ID_COLUMN].values
    df_model = df.drop(columns=[ID_COLUMN], errors="ignore")

    predict_proba, MODEL_NAME, MODEL_VERSION, RUN_ID = process_predict(df_model)

    training_ids = get_training_ids(RUN_ID)

    used_in_training = [1 if cid in training_ids else 0 for cid in ids]

    threshold = 0.5
    predictions = (predict_proba >= threshold).astype(int)

    if save:
        df_output = pd.DataFrame({
            "id": ids,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predict_proba": predict_proba,
            "prediction": predictions,
            "threshold": threshold,
            "used_in_training": used_in_training, 
            "created_at": datetime.now(timezone.utc)
        })

        save_prediction(df_output)

    result = [
        {
            ID_COLUMN: cid,
            "predict_proba": float(proba),
            "prediction": int(pred),
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "threshold": threshold
        }
        for cid, proba, pred in zip(ids, predict_proba, predictions)
    ]

    return {"predictions": result}

@main.get("/drift")
def drift():
    result = run_drift()
    return result

@main.get("/drift/html", response_class=HTMLResponse)
def drift_html():
    html = run_drift_html()
    return HTMLResponse(content=html)