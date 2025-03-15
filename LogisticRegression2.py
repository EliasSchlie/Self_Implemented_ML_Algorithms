import numpy as np
import random 
from preprocessing import train_test_split, StandardScaler
from sklearn.datasets import load_iris


def main():

    iris = load_iris()
    X = iris.data[:, :2]
    y = (iris.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random.randint(0,1000))


    clas = LogRegressor()

    clas.fit(X_train, y_train, alpha=0.2, epochs=50000, X_test=X_test, y_test=y_test, verbosity=5000)

    for i in range(8):
        print(y_test[i], " → ", clas.predict(X_test[i]))

class LogRegressor:
    def __init__(self):
        self.w = None
        self.b = 0
        self.scaler = StandardScaler()


    def predict(self, X, scaled=False):
        if not scaled:
            X = self.scaler.scale_X(X)
        y = 1/(1+np.exp(np.dot(X, self.w) + self.b))
        return y
    
    def fit(self, X, y, alpha=0.01, epochs=100, verbosity=50, X_test=False, y_test=False):
        self.scaler.fit(X,y)
        X = self.scaler.scale_X(X)
        self.w = np.random.rand(X.shape[1])
        for epoch in range(epochs):
            y_pred = self.predict(X, True)
            y_diff = y - y_pred

            dw = np.dot(X.T, y_diff)
            db = np.mean(y_diff)

            self.w = self.w - alpha * dw
            self.b = self.b - alpha * db

            if epoch % verbosity == 0:
                loss = np.mean(np.square(y_diff))
                extre = []
                if all((np.any(X_test), np.any(y_test))):
                    extra = ("\tTest Loss: ", self.loss(X_test,y_test))
                print("Epoch: ", epoch, "\tLoss: ", loss, *extra)

    def loss(self, X, y, scaled=False):
        y_pred = self.predict(X, scaled)
        loss = np.mean(np.square(y - y_pred))
        return loss

if __name__ == "__main__":
    main()