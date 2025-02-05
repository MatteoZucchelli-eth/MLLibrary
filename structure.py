import numpy as np
    
class Network:
    def __init__(self, layers = []):
        self.layers = layers

    def __str__(self):
        return '\n'.join([str(layer) for layer in self.layers])
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, X, y, epochs, learning_rate, batch_size):
        for _ in range(epochs):
            # Shuffle the dataset
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            batches = self.create_batches(X_shuffled, y_shuffled, batch_size)
            for batch in batches:
                X_batch, y_batch = batch
                self.feedforward(X_batch)
                self.backpropagation(y_batch)
                self.gradient_descend(learning_rate)
            print(f"Error: {np.mean((y - self.feedforward(X))**2)}")


    
    def feedforward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            layer.activations = X
        return X
    
    def backpropagation(self, y):
        # Compute the deltas for the final layer
        self.layers[-1].compute_final_deltas(y)
        # Backpropagate the deltas through the network
        for i in range(len(self.layers) - 2, 0, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            current_layer.backward(next_layer.deltas, next_layer.weights)

    def create_batches(self, X, y, batch_size):
        batches = []
        n_data = X.shape[0]
        n_batches = int(np.floor(n_data / batch_size))
        index = 0
        for _ in range(0, n_batches):
            batch = ((X[index:index + batch_size, :], y[index:index + batch_size, :]))
            batches.append(batch)
            index += batch_size
        return batches
    
    def gradient_descend(self, learning_rate):
        for i in range(len(self.layers) - 1):
            current_layer_activations = self.layers[i].activations
            next_layer_deltas = self.layers[i + 1].deltas
            adjustments = current_layer_activations.T @ next_layer_deltas / current_layer_activations.shape[0]
            self.layers[i+1].weights -= adjustments * learning_rate



