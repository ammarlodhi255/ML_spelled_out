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


    def predict(self, X):
        ''' 
        Hypothesis function for our multiple linear regression. 

        Args:
            X (ndarray): Shape((m, n)) m input vectors of size n.

        Returns:
            y_hat (ndarray): Shape((m,)) m predicted values.
        '''
        y_hat = np.dot(X, self.w) + self.b 
        return y_hat
    

    def compute_cost(self):
        ''' 
        Computes cost for all m examples. 

        Returns:
            cost (float): computed cost.
        '''

        y_hat = self.predict(self.X)
        cost = (1/(2*self.m)) * np.sum((y_hat - self.y)**2)
        return cost 


    def compute_gradients(self):
        ''' 
        Computes gradients for model parameters.

        Returns: 
            dj_dw (ndarray): Shape(n,) and dj_db (float): Computed gradients for vector w and the scalar b. 
        '''
        dj_dw = np.zeros((self.n,))
        dj_db = 0.0 
        y_hat = self.predict(self.X)

        for i in range(self.n):
            dj_dw[i] = (1/self.m) * np.sum((y_hat - self.y) * self.X[:, i])

        dj_db = (1/self.m) * np.sum(y_hat - self.y)

        return dj_dw, dj_db



    def fit(self, lr=0.001, epochs=100):
        for i in range(epochs):
            dj_dw, dj_db = self.compute_gradients()
            self.w = self.w - lr*dj_dw 
            self.b = self.b - lr*dj_db
            

if __name__ == "__main__":
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])
    # b_init = 785.1811367994083
    # w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


    lr = LinearRegression(X_train, y_train)
    lr.fit(lr=5e-7)

    print(lr.predict(X_train))

    
