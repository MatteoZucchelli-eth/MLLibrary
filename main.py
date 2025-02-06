import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

from structure import Network
from layers import standard_layer, input_layer, output_layer

# Define PyTorch Model (same architecture as your NumPy model)
class PyTorchNN(nn.Module):
    def __init__(self, input_size):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Linear output
        return x

def train_pytorch_model(model, X_train, y_train, epochs=1000, lr=0.01, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.squeeze())
            loss.backward()
            optimizer.step()

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

    # ----- NumPy Neural Network -----
    net = Network()
    net.add_layer(input_layer(X_train.shape[1], name="Input"))
    net.add_layer(standard_layer(32, "relu", X_train.shape[1], name="Hidden1"))
    net.add_layer(standard_layer(16, "relu", 32, name="Hidden2"))
    net.add_layer(output_layer(1, "linear", 16, name="Output"))  # Linear output for regression

    start_time = time.time()
    net.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=32)
    numpy_training_time = time.time() - start_time
    y_pred_numpy = net.feedforward(X_test)

    # ----- PyTorch Neural Network -----
    model = PyTorchNN(input_size=X_train.shape[1])

    start_time = time.time()
    train_pytorch_model(model, X_train, y_train, epochs=1000, lr=0.01, batch_size=32)
    pytorch_training_time = time.time() - start_time

    with torch.no_grad():
        y_pred_pytorch = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Evaluate both models
    mse_numpy = mean_squared_error(y_test, y_pred_numpy)
    mse_pytorch = mean_squared_error(y_test, y_pred_pytorch)

    print(f"NumPy Neural Network - MSE: {mse_numpy:.4f}, Training Time: {numpy_training_time:.2f} sec")
    print(f"PyTorch Neural Network - MSE: {mse_pytorch:.4f}, Training Time: {pytorch_training_time:.2f} sec")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred_numpy, alpha=0.5, label="NumPy Predictions", color="blue")
    plt.scatter(y_test, y_pred_pytorch, alpha=0.5, label="PyTorch Predictions", color="green")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
