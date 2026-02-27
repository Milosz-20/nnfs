import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# Create layer 1
dense1 = Layer_Dense(2, 3)
# Create ReLU activation for layer 1
activation1 = Activation_ReLU()

# Create layer 2
dense2 = Layer_Dense(3, 3)
# Create Softmax activation for layer 2
activation2 = Activation_Softmax()

# Forward pass with training data through the layer
dense1.forward(X)
# Forward pass through activation func
activation1.forward(dense1.output)

# Forward pass with activation1 output through the layer
dense2.forward(activation1.output)
# Forward pass through activation func
activation2.forward(dense2.output)

print(activation2.output[:5])