import numpy as np

class error:
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
    
se = error("Squared Error", 
           lambda labels, activations: np.mean((activations - labels)**2), 
           lambda labels, activations: 2 * (activations - labels)) 
errors = {
    "se" : se
}