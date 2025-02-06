import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

from structure import Network
from layers import standard_layer, input_layer, output_layer

def main():


    # Load a real-world regression dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    y = y.reshape(-1, 1)  # Ensure correct shape

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling (important for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define your neural network
    net = Network()
    net.add_layer(input_layer(X_train.shape[1], name="Input"))
    net.add_layer(standard_layer(32, "relu", X_train.shape[1], name="Hidden1"))
    net.add_layer(standard_layer(16, "relu", 32, name="Hidden2"))
    net.add_layer(output_layer(1, "linear", 16, name="Output"))  # Linear output for regression

    # Train the model
    net.train(X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32)

    # Predictions
    y_pred_nn = net.feedforward(X_test)

    # Benchmark with Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Evaluate both models
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    print(f"Neural Network MSE: {mse_nn:.4f}")
    print(f"Linear Regression MSE: {mse_lr:.4f}")

    # Plot results
    plt.scatter(y_test, y_pred_nn, alpha=0.5, label="NN Predictions")
    plt.scatter(y_test, y_pred_lr, alpha=0.5, label="LR Predictions", color="red")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

