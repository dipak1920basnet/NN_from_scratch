import numpy as np 
from derivate_activation import function_derivate

def Outer_layer():
    dZ_last = {
        "Sigmoid":Sigmoid,
        "Softmax":Sigmoid,
        "Linear":Sigmoid,
        "Tanh":Tanh,
        "Relu":ReLu,
        "Leakyrelu":Leaky_Relu
    }

    return dZ_last

def Sigmoid(A,Y):
    return A-Y.reshape(-1,1)

def Tanh(A,Y):
    return Sigmoid(A,Y)*(1-np.square(A))

def ReLu(A,Y,Z):
    dZ= np.where(Z>=0,1.0,0.0)
    return (A-Y)*dZ

def Leaky_Relu(A,Y,Z):
    d_g_Z = function_derivate()["Leakyrelu"](Z)
    return (A-Y)*d_g_Z


if __name__ == "__main__":
    Outer_layer()