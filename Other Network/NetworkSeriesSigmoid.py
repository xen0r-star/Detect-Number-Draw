import numpy as np
import sys

DataInput = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
DataOutput = np.array([[5], [6], [7], [8]])

input_dim = DataInput.shape[1]
hidden_dim1 = 32
output_dim = 1

learning_rate = 0.001
learning = 10000

weights_hidden1 = np.random.randn(input_dim, hidden_dim1)
bias_hidden1 = np.zeros((1, hidden_dim1))
weights_output = np.random.uniform(size=(hidden_dim1, output_dim))
bias_output = np.random.uniform(size=(1, output_dim))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for iteration in range(learning):
    hidden_layer_activation1 = np.dot(DataInput, weights_hidden1) + bias_hidden1 # Couche cache 1
    hidden_layer_output1 = sigmoid(hidden_layer_activation1)
    
    output_layer_activation = np.dot(hidden_layer_output1, weights_output) + bias_output
    output_layer_output = sigmoid(output_layer_activation)
    
    error = np.abs(DataOutput - output_layer_output)
    loss = np.mean(error**2)
    
    d_output = error * sigmoid_derivative(output_layer_output)

    error_hidden_layer1 = d_output.dot(weights_output.T)
    d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer_output1)
    
    weights_hidden1 += DataInput.T.dot(d_hidden_layer1) * learning_rate
    bias_hidden1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * learning_rate

    weights_output += hidden_layer_output1.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    sys.stdout.write('\rLearning : {} | Loss : {:.15f}'.format(iteration, loss))
    sys.stdout.flush()

def PredictingResults(value):
    hidden_layer_activation1 = np.dot(value, weights_hidden1) + bias_hidden1
    hidden_layer_output1 = sigmoid(hidden_layer_activation1)

    output_layer_activation = np.dot(hidden_layer_output1, weights_output) + bias_output
    predicted_output = output_layer_activation

    return predicted_output

print("\nPrédiction pour [-52, -51, -50]:", int((PredictingResults([-52, -51, -50])[0][0])))
