import pickle
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------------------

# Resolve path dos artefatos
current_dir = Path(__file__).parent

if (current_dir.parent / "artifacts").exists():
    artefacts_path = current_dir.parent / "artifacts"
else:
    artefacts_path = Path("/app/artifacts")

# ----------------------------------------------------------------------------------------

# Load preprocessor and model
with open(artefacts_path / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open(artefacts_path / "model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------------------------------------------------------------------

def process_test(df: pd.DataFrame) -> pd.DataFrame:

    X = df.copy()

    X_processed = preprocessor.transform(X)

    return X_processed


def process_predict(df) -> list:

    input_df = pd.DataFrame(df)

    processed_df = process_test(input_df)

    prediction = model.predict_proba(processed_df)[:, 1].tolist()

    return prediction