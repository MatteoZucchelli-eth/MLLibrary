import numpy as np

class activation_function:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative

    def __str__(self):
        return f"{self.name}"
    
    def __call__(self, x):
        return self.function(x)
    
    def derivative(self, x):
        return self.derivative(x)
    
def sigmoid(x):
    x = np.clip(x, -500, 500) 
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# ...existing code...

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

activation_functions = {
    "sigmoid" : activation_function("Sigmoid", sigmoid, sigmoid_derivative),
    "relu" : activation_function("ReLU", relu, relu_derivative),
    "tanh" : activation_function("Tanh", tanh, tanh_derivative),
    "softmax" : activation_function("Softmax", softmax, softmax_derivative)
}