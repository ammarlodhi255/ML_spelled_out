import numpy as np 
import matplotlib.pyplot as plt 

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
        return -(1/m) * np.sum(losses)
    

    def fit(self, X, y):
        m, n = X.shape
        dj_dw = np.zeros((n,))
        cost_history = []

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
            
            cost = self._compute_cost(X, y)
            # print(f"Cost: {cost}")
            cost_history.append(cost)

        self._plot_learning_curve(cost_history=cost_history)


    def _plot_learning_curve(self, cost_history):
        plt.plot(np.arange(1, self.iterations+1), cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

if __name__ == "__main__":
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    lr = LogisticRegression(lr = 0.001, iterations=10000)
    lr.fit(X_train, y_train)


