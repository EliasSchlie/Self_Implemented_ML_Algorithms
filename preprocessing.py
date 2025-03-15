import random
import numpy as np

def train_test_split(X,y, test_size=0.2, random_state=42):
    """
    Returns: X_train, X_test, y_train, y_test
    """    
    np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_size = int(test_size * len(indices))
    test_idx, train_idx = indices[:test_size], indices[test_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test

class StandardScaler:
    def __init__(self):
        self.X_means = 0
        self.X_sd = 1
        self.y_mean = 0
        self.y_sd = 1


    def fit(self, X, y):
        self.X_means = np.mean(X, axis=0)
        self.X_sd = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_sd = np.std(y)

    def scale_X(self, X):
        out = (X - self.X_means)/self.X_sd
        return out
    
    def scale_y(self, y):
        out = (y - self.y_mean)/self.y_sd
        return out
    
    def deScale_X(self, X):
        out = (X * self.X_sd) + self.X_means
        return out

    def deScale_y(self, y):
        out = (y * self.y_sd) + self.y_mean
        return out
