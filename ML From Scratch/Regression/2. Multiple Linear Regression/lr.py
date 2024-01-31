import numpy as np 

class LinearRegression():
    def __init__(self, lr=0.01, iterations=100):
        self.lr = lr
        self.iterations = iterations

    
    def _compute_cost(self, X, y):
        m, n = X.shape 
        y_hat = self.predict(X)
        return (1/(2*m)) * np.sum((y_hat - y)**2)
    
        
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros((n,))
        self.b = 0.0

        dw = np.zeros((n,))
        db = 0.0 

        for it in range(self.iterations):
            y_hat = self.predict(X)

            for i in range(n):
                dw[i] = (1/m) * np.sum((y_hat - y) * X[:, i])

            db = (1/m) * np.sum((y_hat - y))

            temp_w = self.w - self.lr * dw
            temp_b = self.b - self.lr * db
            self.w = temp_w 
            self.b = temp_b

            print(f"Cost at Iteration {it}: {self._compute_cost(X, y)}")
        return self


    def predict(self, X):
        return np.dot(X, self.w) + self.b
    

if __name__ == "__main__":
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    regressor = LinearRegression(lr=0.0000001, iterations=1000)

    regressor.fit(X_train, y_train)

    predictions = regressor.predict(X_train)

    print(f"Predictions: {predictions}")
    print(f"y_train: {y_train}")



    