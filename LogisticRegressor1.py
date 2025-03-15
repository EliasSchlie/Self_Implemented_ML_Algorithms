import numpy as np
import random
from preprocessing import StandardScaler, train_test_split
from sklearn.datasets import load_iris



class Clasifier: # Logistic regressor
    def __init__(self, dimensions):
        self.w = np.random.rand(dimensions) # Weights
        self.b = random.random() # Bias
        self.scaler = StandardScaler()

    def predict(self, X: np.ndarray, scaled=False) -> np.ndarray:
        """
        X : np.ndarray, shape (m, n)
            - m = number of examples
            - n = number of features

        Returns: np.ndarray, shape (m,)
            Predicted class for each input example.
        """
        if not scaled:
            X = self.scaler.scale_X(X)
        out = 1/(1 + np.exp(-(np.dot(X, self.w) + self.b)))
        return(out)
        
    def fit(self, X, y, alpha=0.01, epochs=50, X_test=None, y_test=None, verbosity=200):
        m = X.shape[0]
        self.scaler.fit(X,y)
        X = self.scaler.scale_X(X)
        for epoch in range(epochs):
            y_pred = self.predict(X, True)

            dw = np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)

            self.w -= alpha * dw
            self.b -= alpha * db

            if epoch % verbosity == 0:
                error = self.error(X, y, True)
                print("Epoch:      ", epoch, "\nLoss:       ", error)
                if X_test is not None and y_test is not None:
                    print("Test error: ", self.error(X_test, y_test))
                print()

    def error(self, X, y, scaled=False): # Logistic loss
        y_pred = self.predict(X, scaled)
        error = np.mean(-y*np.log(y_pred) - (1-y) * np.log(1-y_pred))
        return(error)
    

iris = load_iris()
X = iris.data[:, :2]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0,1000))


clas = Clasifier(X.shape[1])

clas.fit(X_train, y_train, alpha=0.1, epochs=50000, X_test=X_test, y_test=y_test, verbosity=5000)

for i in range(8):
    print(y_test[i], " → ", clas.predict(X_test[i]))
    