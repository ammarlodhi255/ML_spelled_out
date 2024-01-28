import numpy as np 

class LogisticRegression():
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr 
        self.iterations = iterations
        self.w = np.zeros((X_train.shape[1],))
        self.b = 0.0


    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return (1 / (1 + np.exp(-z)))

    def _compute_cost(self, X, y):
        y_hat = self.predict(X)
        m = X.shape[0]
        losses = (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return (1/m) * np.sum(losses)
    

    def fit(self, X, y):
        m, n = X.shape

        for _ in range(self.iterations):
            y_hat = self.predict(X)
            for i in range(n):
                dj_dw = (1/m) * (np.sum((y_hat - y) * X[:, i]))
            dj_db = (1/m) * np.sum(y_hat - y)

            temp_w = self.w - self.lr * dj_dw
            temp_b = self.b - self.lr * dj_db
            self.w = temp_w 
            self.b = temp_b 

            print(f"Cost: {self._compute_cost(X, y)}")

if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.randint(10, 1000, size=(100, 8)).astype("float64")
    y_train = np.random.randint(0, 2, 100)

    lr = LogisticRegression(lr = 0.000001)
    
    lr.fit(X_train, y_train)


