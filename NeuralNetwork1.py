import numpy as np
import random
from preprocessing import StandardScaler, train_test_split
from sklearn.datasets import load_iris


def main():
    X, y = load_iris(return_X_y=True)
    
    # One hot encoding
    num_classes = len(set(y))
    y = np.array([[1 if label == i else 0 for i in range(num_classes)] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    layers = [
        Layer(4, 4, sigmoid, d_sigmoid),
        Layer(4, 4, sigmoid, d_sigmoid),
        Layer(4, 3, sigmoid, d_sigmoid),
    ]

    network = NN(layers)


    network.fit(X_train, y_train, alpha=0.01, epochs=1000)

    for i in range(8):
        print(y_test[i], " → ", network.predict(X_test[i]))

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.scaler = StandardScaler()

    def predict(self, X, scaled=False):
        if scaled:
            out = X
        else:
            out = self.scaler.scale_X(X)
        for layer in self.layers:
            out = layer.transform(out)
        return out
    
    def fit(self, X, y, alpha=0.01, epochs=50, X_test=None, y_test=None, verbosity=200):
        m = X.shape[0]
        self.scaler.fit(X,y)
        X = self.scaler.scale_X(X)
        for epoch in range(epochs):
            y_prev = y - self.predict(X, True)

            for layer in reversed(self.layers):
                y_prev = layer.backprop(y_prev, alpha)
                
            if epoch % verbosity == 0:
                error = self.error(X, y, True)
                print("Epoch:      ", epoch, "\nLoss:       ", error)
                if X_test is not None and y_test is not None:
                    print("Test error: ", self.error(X_test, y_test))
                print()
    
    def error(self, X, y, scaled=False): # Logistic loss
        y_pred = self.predict(X, scaled)

        epsion = 1e-10
        y_pred = np.clip(y_pred, epsion, 1-epsion)

        error = np.mean(-y*np.log(y_pred) - (1-y) * np.log(1-y_pred))
        return(error)
    

class Layer:
    def __init__(self, inputs=1, neurons=1, g=np.tanh, dg=1):
        self.g = g
        self.dg = dg
        self.W = np.random.rand(inputs, neurons)
        self.b = np.random.rand(neurons)

    def transform(self, X):
        Z = np.dot(X, self.W) + self.b
        out = np.vectorize(self.g)(Z)
        self.X = X # Store last input
        return out
    
    def backprop(self, d_prev, alpha):
        Z = np.dot(self.X, self.W) + self.b
        d_act = np.vectorize(self.dg)(Z)
        d_prev *= d_act

        dW = np.dot(self.X.T, d_prev)
        db = np.mean(d_prev, axis=0)
        d_pass = np.dot(d_prev, self.W.T)

        self.W -= alpha * dW
        self.b -= alpha * db
        
        return d_pass
    
    def derivative(self, d_prev):
        dw = d_prev * self.dg * self.X
        return


def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def d_sigmoid(z):
    return 1





if __name__ == "__main__":
    main()