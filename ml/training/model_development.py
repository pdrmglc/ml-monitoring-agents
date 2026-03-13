# %% Imports
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from ml.preprocessing.preprocessor import Preprocessor

# %% Load data
df = pd.read_csv("../../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

target = "Churn"
id_col = "customerID"

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42
)

X_train = df_train.drop(columns=[target, id_col])
y_train = df_train[target].map({"Yes": 1, "No": 0})

# %% Identify categorical features
categorical_features = X_train.select_dtypes(include=["object", "str"]).columns.tolist()

# ----------------------------------------------------------------------------------------

# %% Apply encoding

preprocessor = Preprocessor(categorical_features)

X_train_processed = preprocessor.fit_transform(X_train, y_train)

# ----------------------------------------------------------------------------------------

# %% Train model
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
)

model.fit(X_train_processed, y_train)
# ----------------------------------------------------------------------------------------

# %% Save encoders and model

with open("../../artifacts/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("../../artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../../data/df_test.csv", "w") as f:
    df_test.to_csv(f, index=False)
# %%
