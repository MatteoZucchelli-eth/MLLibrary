import numpy as np
from activationFunctions import activation_functions as afs
from errors import errors

# every layer has the weights previous to him

class input_layer:
    def __init__(self, n_nodes, name = None):
        self.name = name
        self.n_nodes = n_nodes
        self.activations = np.zeros((self.n_nodes, 1))

    def __str__(self):
        return f"Input Layer {self.name}: {self.n_nodes}"
    
    def forward(self, X):
        return X

    
class standard_layer:
    def __init__(self, n_nodes, activation, input_size, name = None):
        self.name = name
        self.n_nodes = n_nodes
        if activation not in afs:
            raise ValueError(f"Activation function {activation} not found")
        self.activation = afs[activation]
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=(input_size, n_nodes))
        self.biases = np.random.randn(self.n_nodes, 1)
        self.activations = np.zeros((input_size, self.n_nodes))
        self.deltas = np.zeros((input_size, 1))

    def __str__(self):
        return f"Standard layer {self.name}: Number of nodes: {self.n_nodes}, Activation: {self.activation}"
    
    def forward(self, X):
        """
        Forward pass a matrix X for this layer
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x = X @ self.weights + self.biases.T # (n_samples, n_nodes)
        return self.activation(x)
    
    def backward(self, next_deltas, next_weights):
        """
        Backward calculate the deltas for this layer
        """
        self.deltas = next_deltas @ next_weights.T * self.activation.derivative(self.activations) # (n_samples, n_nodes)
        
     
class output_layer(standard_layer):
    def __init__(self, size, activation, input_size, name = None, error = "se"):
        super().__init__(size, activation, input_size, name)
        self.error = errors[error]
    
    def compute_final_deltas(self, y):
        d_error = self.error.derivative(y, self.activations) # (n_samples, n_nodes)
        d_function = self.activation.derivative(self.activations) # (n_samples, n_nodes)
        self.deltas = d_error * d_function


    
class pooling_layer:
    def __init__(self, size, stride, dimension, name = None):
        self.name = name
        self.size = size
        self.stride = stride
        self.dimension = dimension

    def __str__(self):
        return f"Pooling Layer {self.name}: {self.size}, Stride: {self.stride}, Dimension: {self.dimension}"

class conv_layer:
    def __init__(self, size, stride, filters, activation):
        self.size = size
        self.stride = stride
        self.filters = filters
        if activation not in afs:
            raise ValueError(f"Activation function {activation} not found")
        self.activation = afs[activation]

    def __str__(self):
        return f"Convolutional Layer: {self.size}, Stride: {self.stride}, Filters: {self.filters}, Activation: {self.activation}"