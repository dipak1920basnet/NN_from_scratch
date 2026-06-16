from activations import activation
import numpy as np 
activation_function = activation()

def function_derivate():
    derivative = {
        "Linear":d_Linear, 
        "Sigmoid":d_Sigmoid,
        "Tanh":d_tanh,
        "Relu":d_relu,
        "Leakyrelu": d_leaky_relu,
    }
    return derivative

def d_Linear(Z):
    # return 1 
    return np.ones_like(Z)

def d_Sigmoid(Z):
    a = activation_function["Sigmoid"](Z)
    return a*(1-a)

def d_tanh(Z):
    a = activation_function["Tanh"](Z)
    return (1-np.square(a))

def d_relu(Z):
    # return 0 if Z < 0 else 1
    # return (Z > 0).astype(float)
    return np.where(Z>=0,1.0,0.0)

def d_leaky_relu(Z):
    # return  0.01 if Z < 0 else 1
    return np.where(Z>=0,1.0,0.01)


if __name__ == "__main__":
    function_derivate()