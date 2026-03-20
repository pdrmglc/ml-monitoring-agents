from pydantic import create_model
from typing import Optional, Union

from ml.schema.data_definition import NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS, ID_COLUMN, BIN_COLUMNS


def create_feature_model():
    fields = {}

    # numéricos
    for col in NUMERICAL_COLUMNS:
        fields[col] = (Optional[float], ...)

    # categóricos
    for col in CATEGORICAL_COLUMNS:
        fields[col] = (Optional[str], ...)
    
    # binários
    for col in BIN_COLUMNS:
        fields[col] = (Optional[Union[int, str]], ...)

    # id
    fields[ID_COLUMN] = (str, ...)

    return create_model("FeatureRow", **fields)


FeatureRow = create_feature_model()