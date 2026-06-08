import numpy as np 
from activations import activation, compute_Z
from neuron import Dense
activation_list = activation()

X = np.matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10,11,12]
])

class Sequential():
    def __init__(self, NN:list):
        self.NN = NN
        self.track_W = []
        self.track_B = []
        self.track_Z = []
        self.track_A = []
        self.track_dW = [None] * len(self.track_W)
        self.track_dB = [None] * len(self.track_B)
        self.track_dZ = [None] * len(self.track_Z)
        self.track_dA = [None] * len(self.track_A)

    def forward_pass(self, X):
        self.track_A.append(X)
        for i in  range(len(self.NN)):
            X = self.track_A[-1]
            print(f"layer_{i}: {X.shape}")
            unit , activations = self.NN[i]

            w = np.random.rand(unit, X.shape[1])
            b = np.random.rand(unit)
            
            Z = compute_Z(w,X,b)
            A = activation_list[activations](Z)

            self.track_W.append(w)
            self.track_B.append(b)
            self.track_Z.append(Z)
            self.track_A.append(A)

    def back_prop(self):
        pass
    
    def gradient_descent(self):
        pass

    def fit(self,X, epochs= 1):
        self.forward_pass(X)


            

model = Sequential([
    Dense(7,"linear"),
    Dense(3,"ReLu"),
    Dense(9,"tanh"),
    Dense(1,"Sigmoid")
])

print(model.NN)


model.fit(X)
