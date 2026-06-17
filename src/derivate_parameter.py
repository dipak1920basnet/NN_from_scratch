import numpy as np

def Derivate_parameter():
    ...

def derivate_Z(W,dZ,dg_Z):
    """
    formula: 
    dZl = (W*dZ)*dg_Z
    """
    # return (np.transpose(W)*dZ)*dg_Z
    return (dZ@W)*dg_Z


def derivate_W(dZl,A,m):
    """
    dW = (1/m)*dZl*AT
    """
    dZl = dZl.reshape(-1,1)
    A = A.reshape(-1,1)
    return (dZl*np.transpose(A))/m

def derivate_b(dZl,m):
    """
    db = (1/m)*np.sum(dZl)
    """
    return np.sum(dZl)/m

if __name__ == "__main__":
    Derivate_parameter()