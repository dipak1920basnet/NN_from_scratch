import numpy as np 
from activations import activation, compute_Z
from neuron import Dense
from derivate_activation import function_derivate
from outer_layer import Outer_layer
from derivate_parameter import derivate_Z, derivate_W, derivate_b

activation_list = activation()
calc_func_derivate = function_derivate()
output_layer_derivate = Outer_layer()

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
        total_layer = len(self.NN)
        self.track_W = []
        self.track_B = []
        self.track_Z = []
        self.track_A = []
        self.track_dW = [None] * total_layer
        self.track_dB = [None] * total_layer
        self.track_dZ = [None] * total_layer
        # self.track_dA = [None] * len(self.track_A)

    def forward_pass(self, X):
        self.track_A.append(X)
        for i in  range(len(self.NN)):
            X = self.track_A[-1]
            # print(f"layer_{i}: {X.shape}")
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
        m = self.track_A[0].shape[0]
        self.track_A = self.track_A[1:]
        def last_layer():
            # get the name of activation function 
            activation_function_name = self.NN[-1][1]
            # use that name to get the derivative of particular activation function. 
            Al = self.track_A[-1] #Al is the activation for the output layer
            dZ = output_layer_derivate[activation_function_name](Al,Y)
            """
            Since Al is the activation for the output layer 
            and Al is self.track_A[-1]
            Al-1 will be self.track_A[-2]
            """
            Al_1 = self.track_A[-2]
            dw = derivate_W(dZl=dZ,A=Al_1,m=m)

            db = derivate_b(dZl=dZ,m=m)
            return dZ, dw, db
        
        dZ, dW, dB = last_layer()
        self.track_dZ[-1] = dZ
        self.track_dW[-1] = dW
        self.track_dB[-1] = dB

        for i in range(len(self.NN)-2,0,-1): 
            """
            i starts with second last layer
            """
            print(i)
            dZ_L_1 = self.track_dZ[i+1]
            # print(dZ_L_1)
            w = np.transpose(self.track_W[i+1])
            # get the name of activation function 
            activation_function_name = self.NN[i][1]
            derivate_g_Z = calc_func_derivate[activation_function_name](self.track_Z[i])
            dZ = derivate_Z(W=w,dZ=dZ_L_1, dg_Z=derivate_g_Z)

            A = np.transpose(self.track_A[i+1])
            dW = derivate_W(dZl=dZ,A=A,m=m)
            dB = derivate_b(dZl=dZ,m=m)
            
            self.track_dZ[i] = dZ
            self.track_dW[i] = dW
            self.track_dB[i] = dB

        # last_layer_dZ = self.track_A[-1]- Y.reshape(-1,1)
        # m = len(self.track_A[0])
        # last_layer_dw = ((last_layer_dZ)*self.track_A[-2].transpose())/m
        # last_layer_db =  np.sum(last_layer_dZ, axis=1, keepdims=True)/m
        # self.track_dZ[-1] = last_layer_dZ
        # self.track_dW[-1] = last_layer_dw
        # self.track_dB[-1] = last_layer_db
        # dZ = self.track_dZ[-1]
        # dW = self.track_dW[-1]
        # dB = self.track_dB[-1]
        # return 
    
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

# print(model.NN)


model.fit(X,Y)
# print(model.track_A[-1])
model.back_prop(Y)
# print()
# print()
# print()


# print(model.track_dW)
# print()
# print()
# print()

# print(model.track_dB)

# print()
# print()
# print()

# # print(model.track_dZ)

# print(len(model.track_A))
# print(len(model.track_Z))