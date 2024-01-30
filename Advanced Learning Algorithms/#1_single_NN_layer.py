import numpy as np 

X_sample = np.array([9, 190])

num_of_neurons = 3

W_1 = np.array([
    [0.001, 0.003],
    [0.0014, 0.0035],
    [0.0051, 0.0053]
])

B_1 = np.array([0.004, 0.001, 0.002])

a_1 = np.zeros((num_of_neurons,))

for i in range(num_of_neurons):
    a_1[i] = np.dot(X_sample, W_1[i]) + B_1[i]


# Final Layer (Output Layer)
W_2 = np.array([0.001, 0.003, 0.002])
b = 0.0071

z = np.dot(a_1, W_2) + b

output = 1 / (1 + np.exp(-z))

print(f"Output: {output}")

