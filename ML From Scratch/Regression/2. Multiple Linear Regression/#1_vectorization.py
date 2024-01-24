import numpy as np
import time

X = np.random.random_sample((20000, 16))
W = np.random.random_sample((16,))
b = 0.1 
X_W = np.zeros((20000,))

start_time = time.time()

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X_W[i] += X[i, j] * W[j]

print("Without Vectorization: --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
X_W_2 = np.dot(X, W)
print("With Vectorization: --- %s seconds ---" % (time.time() - start_time))


print(X_W)
print(X_W_2)