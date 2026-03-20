# ml/schema/data_definition.py

ID_COLUMN = "customerID"

TARGET_COLUMN = "Churn"

NUMERICAL_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BIN_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "Contract",
    "PaymentMethod",
]

ALL_COLUMNS = NUMERICAL_COLUMNS + BIN_COLUMNS + CATEGORICAL_COLUMNS + [ID_COLUMN, TARGET_COLUMN]