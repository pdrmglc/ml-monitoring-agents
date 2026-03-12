import pandas as pd

df = pd.read_csv("app/data/churn_dataset_full.csv")

train = df[df["month"] <= 6]
validation = df[(df["month"] > 6) & (df["month"] <= 8)]
test = df[df["month"] > 8]

train.to_csv("app/data/train.csv", index=False)
validation.to_csv("app/data/validation.csv", index=False)
test.to_csv("app/data/test.csv", index=False)

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)