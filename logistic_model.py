import numpy as np


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