import numpy as np 
import copy, math 

class LinearRegression():
    def __init__(self, X, y):
        self.X = X 
        self.y = y 
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.w = np.zeros((self.n,))
        self.b = 0.0 


    def predict(X, w, b):
        ''' 
        Hypothesis function for our multiple linear regression. 

        Args:
            X (ndarray): Shape((m, n)) m input vectors of size n.
            w (ndarray): Shape((n,)) n model parameters for corresponding input features.
            b (float): model parameter.

        Returns:
            y_hat (ndarray): Shape((m,)) m predicted values.
        '''
        y_hat = np.dot(X, w) + b 
        return y_hat
    
