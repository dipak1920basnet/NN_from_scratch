import numpy as np 

def activation():
    activation_list = {
        "Linear":Linear,
        "Relu":ReLU,
        "Sigmoid":Sigmoid,
        "Tanh":Tanh,
        "Leakyrelu":LeakyReLU
    }
    return activation_list

def compute_Z(W,X,b):
    if X.shape[1] == W.shape[1]:
        return np.matmul(X,W.transpose()) + b.reshape(1,-1)
    raise ValueError(
        f"Shape mismatch: X {X.shape}, W {W.shape}")

def Linear(Z):
    return Z

def ReLU(Z):
    g_Z = np.maximum(0,Z)
    return g_Z 

def Sigmoid(Z):
    g_Z = 1/(1+np.exp(-Z))
    return g_Z

def Tanh(Z):
    # Z1 = np.exp(Z)
    # Z2 = np.exp(-Z)

    # g_Z = (Z1 - Z2) / (Z1 + Z2)
    g_Z = np.tanh(Z)
    return g_Z

def LeakyReLU(Z):
    g_Z = np.maximum(0.01*Z,Z)
    return g_Z

def softmax(Z):
    pass

if __name__ == "__main__":
    activation()