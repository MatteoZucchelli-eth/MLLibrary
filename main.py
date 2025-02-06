import numpy as np
from structure import Network
from layers import standard_layer, input_layer, pooling_layer, conv_layer, output_layer

def main():
    net = Network()
    net.add_layer(input_layer(2, name="Input"))
    net.add_layer(standard_layer(4, "relu", 2, name="Hidden"))
    net.add_layer(standard_layer(5, "relu", 4, name="Hidden"))
    net.add_layer(output_layer(1, "linear", 5, name="Output"))

    # Generate data
    X = np.ones((1000, 2))
    y = -5 * 0.3 * np.ones((1000, 1))

    # Training with error monitoring
    errors = []
    for epoch in range(100):  # Increased epochs
        initial_pred = net.feedforward(X)
        net.train(X, y, epochs=1, learning_rate=0.1, batch_size=32)  # Adjusted hyperparameters
        final_pred = net.feedforward(X)
        error = np.mean((y - final_pred)**2)
        errors.append(error)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Error: {error}")
            print(f"Predictions: {final_pred[:5]}")
            print(f"Target: {y[:5]}")

            
if __name__ == "__main__":
    main()

