
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from logisticRegrassionPipeline import LogisticRegScratch

# 1. Load the separate testing dataset
test_data = pd.read_csv("Dataset/new_stockout_test_dataset.csv")

features = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "rolling_7d_sales",
    "rolling_14d_sales",
    "inventory_to_demand_ratio",
    "sales_trend",
    "stock_pressure_ratio",
    "Price",
    "Discount",
    "Competitor Pricing"
]

# 2. Prepare testing data
test_data = test_data.replace([np.inf, -np.inf], np.nan)
test_data = test_data.dropna(
    subset=features + ["stockout_risk"]
)

X_test = test_data[features].to_numpy(dtype=float)
y_test = test_data["stockout_risk"].to_numpy(dtype=int)

# 3. Load the saved model
model = LogisticRegScratch()
model.load_model("logistic_model.npz")

# 4. Load scaling values saved during training
scaler = np.load("scaler.npz")
mean = scaler["mean"]
std = scaler["std"]

# 5. Apply the SAME training scaling
X_test = (X_test - mean) / std

# 6. Predict
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

print("First 10 predictions:", predictions[:10])
print("First 10 probabilities:", probabilities[:10])

# 7. Evaluate
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 score:", f1_score(y_test, predictions, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
