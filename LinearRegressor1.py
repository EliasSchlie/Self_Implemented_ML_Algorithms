import numpy as np
import random
from sklearn.datasets import fetch_california_housing
from preprocessing import train_test_split, StandardScaler

class LinearRegressor:
    def __init__(self, dimensions):
        self.w = np.random.rand(dimensions) # Weights
        self.b = random.random() # Bias
        self.scaler = StandardScaler()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : np.ndarray, shape (m, n)
            - m = number of examples
            - n = number of features

        Returns: np.ndarray, shape (m,)
            Predicted values for each input example.
        """
        X = self.scaler.scale_X(X)
        out = np.dot(X, self.w) + self.b
        out = self.scaler.deScale_y(out)
        return(out)
        
    def fit(self, X, y, alpha=0.01, epochs=50, X_test=None, y_test=None, verbosity=200):
        m = X.shape[0]
        self.scaler.fit(X,y)
        X_real, y_real = X,y
        X, y = self.scaler.scale_X(X), self.scaler.scale_y(y)
        for epoch in range(epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.w -= alpha * dw
            self.b -= alpha * db

            if epoch % verbosity == 0:
                error = self.error(self.scaler.deScale_X(X), self.scaler.deScale_y(y))
                print("Epoch:      ", epoch, "\nMSE:        ", error)
                if X_test is not None and y_test is not None:
                    print("Test error: ", self.error(X_test, y_test))
                print()

    def error(self, X, y): # Squared error loss
        error = (1/(X.shape[0])) * np.sum(np.square(self.predict(X) - y))
        return(error)
    



data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y)

reg = LinearRegressor(X.shape[1])

reg.fit(X_train, y_train, alpha=0.05, epochs=5000, X_test=X_test, y_test=y_test, verbosity=1000)

for i in range(8):
    print(y_train[i], " → ", reg.predict(X_train[i]))