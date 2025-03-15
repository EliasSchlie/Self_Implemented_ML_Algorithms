import numpy as np
import pandas as pd



def main():
    X = np.array([
        [1, 0, 1],  
        [0, 1, 0],  
        [1, 1, 1],  
        [0, 0, 1],  
        [1, 0, 0],  
        [0, 1, 1],  
        [1, 1, 0],  
        [0, 0, 0],  
        [1, 0, 1],  
        [0, 1, 0]   
    ])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    tre = Tree()
    tre.fit(X,y)
    print(tre.predict(X))
    print(tre.root)

class Tree:
    def __init__(self):
        self.root = None

    def predict(self, X, indices=None, node=None, y=None):
        if node is None:
            node = self.root
        if indices is None:
            indices = np.arange(X.shape[0])
        if y is None:
            y = np.zeros(X.shape[0])
        if node.prediction is not None:
            y[indices] = node.prediction
            return y
        r_indices = indices[X[indices, node.feature] == 1]
        l_indices = indices[X[indices, node.feature] == 0]
        y = self.predict(X, r_indices, node.right, y)
        y = self.predict(X, l_indices, node.left, y)
        return y
    
    def fit(self, X, y, continuous=False):
        self.root = TreeNode(X, y, np.arange(X.shape[0]), continuous)
            

class TreeNode:
    def __init__(self, X, y, indices, continuous):
        if len(indices) <= 1:
            self.prediction = y[indices]
        elif np.all(y[indices] == y[indices][0]):
            self.prediction = y[indices][0]
        else:
            self.prediction = None
            lowest = None
            for i in range(X.shape[1]): # For all possible splits
                right = indices[X[indices,i] == 1]
                left = indices[X[indices,i] == 0]
                if len(right) == 0 or len(left) == 0:
                    continue
                if continuous:
                    pass
                entropy = 0
                for side in (left,right):
                    p = len(y[side] == 1)
                    entropy = len(side) * self.h(p)
                if lowest is None or entropy < lowest:
                    lowest = entropy
                    self.feature = i
            r_in = indices[X[indices,self.feature] == 1]
            l_in = indices[X[indices,self.feature] == 0]
            self.right = TreeNode(X,y,r_in,continuous)
            self.left = TreeNode(X,y,l_in,continuous)

    def h(self, p):
        return -p * np.log2(p) - (1-p) * np.log2(p)


    def __str__(self, depth=0):
        indent = "  " * depth  # Create indentation based on depth level
        if self.prediction is not None:
            return f"{indent}Leaf: Predict {self.prediction}"
        
        result = f"{indent}Feature {self.feature}:\n"
        
        if self.left:
            result += f"{indent}  Left -> {self.left.__str__(depth + 1)}\n"
        if self.right:
            result += f"{indent}  Right -> {self.right.__str__(depth + 1)}"
        
        return result.strip()



if __name__ == "__main__":
    main()