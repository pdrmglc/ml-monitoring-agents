from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from .model_inference import process_predict


class InputData(BaseModel):
    features: List[Dict[str, Any]]


main = FastAPI()


@main.post("/predict")
def predict(data: InputData):

    pred = process_predict(data.features)

    return {"prediction": pred}