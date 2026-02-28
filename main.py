import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        # Initialize random weights and no biases (0)
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
        # Get raw unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample (0-1)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    # Calculates the data and regularization losses (how much model is wrong)
    # Based on output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in the batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # The goal is to extract probabilities for the target classes only
        # Probabilities for target values - categorical/sparse labels (e.g. [0,2,1])
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Probabilities for mask values - one-hot vectors (e.g. [[1,0,0], [0,0,1], [0,1,0]])
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Create dataset
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

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Forward pass with training data through the layer
dense1.forward(X)
# Forward pass through activation func
activation1.forward(dense1.output)

# Forward pass with activation1 output through the layer
dense2.forward(activation1.output)
# Forward pass through activation func
activation2.forward(dense2.output)

print(activation2.output[:5])

# Forward pass through activation func
# Takes output of second dense layer and returns loss
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)

# Calculate accuracy from activation2 output and target values
# Example: if model was right 2 out of 3 times, accuracy equals to 0.66 (2/3)
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print("acc: ", accuracy)