from activations import activation, compute_Z
import numpy as np 

activation_list = activation()

X = np.matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10,11,12]
])
def Dense(unit=2,activation="linear"):
    activation = activation.capitalize()
    weights = np.random.rand(unit,X.shape[1])
    bias = np.random.rand(unit)
    Z = compute_Z(X,weights,bias)
    A = activation_list[activation](Z)
    return weights,bias,Z,A

weight, bias, Z,A = Dense()
print(f"Weight: {weight}\n\n")
print(f"Bias: {bias} \n\n")
print(f"Z: {Z}")
print(f"A: {A}")

