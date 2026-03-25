import numpy as np
import pandas as pd
from app.model_inference import process_predict, save_prediction, get_training_ids, create_connected_engine
from ml.schema.data_definition import ID_COLUMN, FEATURE_COLUMNS

def run_batch_prediction():

    engine = create_connected_engine()

    # 1. Carrega dados
    for chunk in pd.read_sql("SELECT * FROM raw_data", engine, chunksize=10000):
        
        ids = chunk[ID_COLUMN].values

        df_model = chunk[FEATURE_COLUMNS]

        # 2. Predição
        predict_proba, MODEL_NAME, MODEL_VERSION, RUN_ID = process_predict(df_model)

        threshold = 0.5
        predictions = (np.array(predict_proba) >= threshold).astype(int)

        # 3. Marca treino
        training_ids = get_training_ids(RUN_ID)

        used_in_training = [
            1 if cid in training_ids else 0
            for cid in ids
        ]

        # 4. Monta output
        df_output = pd.DataFrame({
            "id": ids,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predict_proba": predict_proba,
            "prediction": predictions,
            "threshold": threshold,
            "used_in_training": used_in_training,
        })

        # 5. Salva
        save_prediction(df_output)


if __name__ == "__main__":
    try:
        run_batch_prediction()
    except Exception as e:
        print(f"Erro durante a predição em lote: {e}")

