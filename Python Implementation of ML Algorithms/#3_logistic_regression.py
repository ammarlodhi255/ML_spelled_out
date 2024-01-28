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
    

    def fit(self, X, y):
        m, n = X.shape
        dj_dw = np.zeros((n,))

        for _ in range(self.iterations):
            y_hat = self.predict(X)
            err = y_hat - y

            for i in range(n):
                dj_dw[i] = (1/m) * (np.sum((err) * X[:, i]))
            dj_db = (1/m) * np.sum(err)

            temp_w = self.w - self.lr * dj_dw
            temp_b = self.b - self.lr * dj_db
            self.w = temp_w 
            self.b = temp_b 
            


if __name__ == "__main__":
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    lr = LogisticRegression(lr = 0.001, iterations=10000)
    lr.fit(X_train, y_train)


