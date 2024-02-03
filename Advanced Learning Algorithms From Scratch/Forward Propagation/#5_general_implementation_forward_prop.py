import numpy as np


def g(z):
    return 1 / (1 + np.exp(-z))


def dense(a_in, W, b):
    units = W.shape[1]
    a_n = np.zeros((W.shape[1], ))

    for i in range(units):
        w = W[:, i]
        out = np.dot(a_in, w) + b[i]
        a_n[i] = g(out)

    return a_n


def vectorized_dense(a_in, W, b):
    # return g(np.dot(a_in, W) + b) # Equivalent return statement
    return g(np.matmul(a_in.T, W) + b)


if __name__ == '__main__':
    W = np.array([
        [1, -3, 5],
        [2, 4, -6]
    ])

    b = np.array([-1, 1, 2])

    a_in = np.array([-2, 4])


    print(dense(a_in, W, b))
    print(vectorized_dense(a_in, W, b))
