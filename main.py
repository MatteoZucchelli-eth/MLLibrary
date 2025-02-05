import numpy as np
from structure import Network
from layers import standard_layer, input_layer, pooling_layer, conv_layer, output_layer

def main():
    net = Network()
    net.add_layer(input_layer(2, name="Input"))
    net.add_layer(standard_layer(4, "relu", 2, name="Hidden"))
    net.add_layer(output_layer(1, "relu", 4, name="Output"))  # Changed to 1 output

    # Simple binary classification
    X = np.ones((1000, 2))
    y = 5 * 0.3 * np.ones((1000, 1))

    print(net.feedforward(X)[:10])
    net.train(X, y, epochs=10, learning_rate=0.01, batch_size=10)
    print(net.feedforward(X)[:10])

if __name__ == "__main__":
    main()

