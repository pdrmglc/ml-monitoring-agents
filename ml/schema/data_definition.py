# ml/schema/data_definition.py

ID_COLUMN = "customer_id"

TARGET_COLUMN = "churn"

NUMERICAL_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
]

BIN_COLUMNS = [
    "senior_citizen",
    "partner",
    "dependents",
    "multiple_lines",
    "phone_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "internet_service",
    "contract",
    "payment_method",
]

ALL_COLUMNS = NUMERICAL_COLUMNS + BIN_COLUMNS + CATEGORICAL_COLUMNS + [ID_COLUMN, TARGET_COLUMN]
FEATURE_COLUMNS = NUMERICAL_COLUMNS + BIN_COLUMNS + CATEGORICAL_COLUMNS