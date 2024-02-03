import numpy as np 

# NN with 1 hidden layer having 3 units.

a_0 = np.array([[9, 190]])

W_1 = np.array([
    [0.001, 0.003],
    [0.0014, 0.0035],
    [0.0051, 0.0053]
])

B_1 = np.array([0.004, 0.001, 0.002])

a_1 = (1 / (1 + np.exp(-(np.dot(a_0, W_1.T) + B_1))))

print(f"a1: {a_1}")


# Final Layer (Output Layer)
W_2 = np.array([0.001, 0.003, 0.002])
b = 0.0071

a_2 = np.dot(a_1, W_2) + b

output = 1 / (1 + np.exp(-a_2))

print(f"Output: {output}")

