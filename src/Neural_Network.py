import numpy as np 
from activations import activation, compute_Z
from neuron import Dense
from derivate_activation import function_derivate

activation_list = activation()
calc_func_derivate = function_derivate()

X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10,11,12]
])

Y = np.array([
    1,2,3,4
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
        # self.track_dA = [None] * len(self.track_A)

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

    def back_prop(self,Y):
        last_layer_dZ = self.track_A[-1]- Y.reshape(-1,1)
        m = len(self.track_A[0])
        last_layer_dw = ((last_layer_dZ)*self.track_A[-2].transpose())/m
        last_layer_db =  np.sum(last_layer_dZ, axis=1, keepdims=True)/m
        self.track_dZ[-1] = last_layer_dZ
        self.track_dW[-1] = last_layer_dw
        self.track_dB[-1] = last_layer_db
        dZ = self.track_dZ[-1]
        dW = self.track_dW[-1]
        dB = self.track_dB[-1]
        return 
    
    def gradient_descent(self):
        pass

    def fit(self,X, epochs= 1):
        self.forward_pass(X)
        for i in epochs:
            self.gradient_descent


            

model = Sequential([
    Dense(7,"linear"),
    Dense(3,"ReLu"),
    Dense(9,"tanh"),
    Dense(1,"Sigmoid")
])

print(model.NN)


model.fit(X,Y)
print(model.track_A[-1])
print(model.back_prop(Y))