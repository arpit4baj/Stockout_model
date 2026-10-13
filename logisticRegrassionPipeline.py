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

# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    r"Dataset\stockout_test_dataset.csv"
)

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Store ID", "Product ID", "Date"]).copy()


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

group_cols = ["Store ID", "Product ID"]

# Previous 7 days of sales
data["rolling_7d_sales"] = (
    data.groupby(group_cols)["Units Sold"]
    .transform(lambda s: s.shift(1).rolling(7).sum())
)

# Previous 14 days of sales
data["rolling_14d_sales"] = (
    data.groupby(group_cols)["Units Sold"]
    .transform(lambda s: s.shift(1).rolling(14).sum())
)

# Previous 7-day sales compared with the 7 days before that
data["previous_7d_sales"] = (
    data.groupby(group_cols)["Units Sold"]
    .transform(lambda s: s.shift(8).rolling(7).sum())
)

data["sales_trend"] = (
    data["rolling_7d_sales"] - data["previous_7d_sales"]
)

# Inventory coverage based on recent average daily sales
avg_daily_sales = data["rolling_7d_sales"] / 7

data["inventory_to_demand_ratio"] = (
    data["Inventory Level"] /
    avg_daily_sales.replace(0, np.nan)
)

# Current sales relative to current stock plus orders
denominator = data["Inventory Level"] + data["Units Ordered"]

data["stock_pressure_ratio"] = (
    data["Units Sold"] /
    denominator.replace(0, np.nan)
)


# ==========================================
# 3. CREATE TARGET
# ==========================================

# Assumption: supplier replenishment lead time = 3 days
lead_time = 3

data["lead_time_demand"] = (
    data.groupby(group_cols)["Demand Forecast"]
    .transform(
        lambda s: s.shift(-1)
                   .rolling(lead_time)
                   .sum()
                   .shift(-(lead_time - 1))
    )
)

# 1 = inventory insufficient for forecast demand
# 0 = inventory sufficient for forecast demand
data["stockout_risk"] = (
    data["Inventory Level"] < data["lead_time_demand"]
).astype(int)

# Remove rows without enough future data to construct the target
data = data.dropna(subset=["lead_time_demand"]).copy()


# ==========================================
# 4. SELECT FEATURES
# ==========================================

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

target = "stockout_risk"

# Remove rows with missing or infinite feature values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=features + [target]).copy()

# Sort chronologically before splitting
data = data.sort_values("Date").copy()


# ==========================================
# 5. TIME-BASED TRAIN / TEST SPLIT
# ==========================================

# Split by unique dates to prevent records from the
# same date appearing in both training and testing.
dates = np.sort(data["Date"].unique())
split_index = int(len(dates) * 0.8)

train_dates = dates[:split_index]
test_dates = dates[split_index:]

train_data = data[data["Date"].isin(train_dates)]
test_data = data[data["Date"].isin(test_dates)]

X_train = train_data[features].to_numpy(dtype=float)
X_test = test_data[features].to_numpy(dtype=float)

y_train = train_data[target].to_numpy(dtype=float)
y_test = test_data[target].to_numpy(dtype=int)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==========================================
# 6. STANDARDIZE FEATURES
# ==========================================

# Learn scaling parameters only from training data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)


std[std == 0] = 1
np.savez("scaler.npz", mean=mean, std=std)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# ==========================================
# 7. LOGISTIC REGRESSION FROM SCRATCH
# ==========================================
class LogisticRegScratch:
    def __init__(self, learn_rate=0.01, epochs=1000):
        self.lr = learn_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        if n_samples == 0 or len(y) != n_samples:
            raise ValueError("Invalid training data.")
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            probabilities = self.sigmoid(z)
            error = probabilities - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            p = np.clip(probabilities, 1e-15, 1 - 1e-15)
            loss = -np.mean(
                    y * np.log(p) +
                (1 - y) * np.log(1 - p)
            )
            self.loss_history.append(float(loss))
        return self
    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained or loaded.")
        X = np.asarray(X, dtype=float)
        return self.sigmoid(X @ self.weights + self.bias)
    def predict(self, X, threshold=0.40):
        return (self.predict_proba(X) >= threshold).astype(int)
    def save_model(self, filename="logistic_model.npz"):
        if self.weights is None:
            raise ValueError("No trained model to save.")
        np.savez(
                filename,
            weights=self.weights,
            bias=self.bias
        )
    def load_model(self, filename="logistic_model.npz"):
        with np.load(filename, allow_pickle=False) as saved:
            weights = saved["weights"]
            bias = float(saved["bias"])
        if not np.all(np.isfinite(weights)) or not np.isfinite(bias):
            raise ValueError("Saved model contains invalid values.")
        self.weights = weights
        self.bias = bias
        return self

# ==========================================
# 8. TRAIN MODEL
# ==========================================

model = LogisticRegScratch(
    learn_rate=0.18,
    epochs=4000
)

model.fit(X_train, y_train)


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nFirst 10 predicted classes:")
print(predictions[:10])

print("\nFirst 10 predicted probabilities:")
print(probabilities[:10])

print("\nFinal training loss:")
print(model.loss_history[-1])


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

print("\nAccuracy:", accuracy_score(y_test, predictions))

print(
    "Precision:",
    precision_score(y_test, predictions, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, predictions, zero_division=0)
)

print(
    "F1 Score:",
    f1_score(y_test, predictions, zero_division=0)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 11. INSPECT LEARNED WEIGHTS
# ==========================================

weights_table = pd.DataFrame({
    "Feature": features,
    "Weight": model.weights
})

weights_table["Absolute Weight"] = (
    weights_table["Weight"].abs()
)

weights_table = weights_table.sort_values(
    "Absolute Weight",
    ascending=False
)

print("\nLearned Feature Weights:")
print(weights_table.to_string(index=False))

