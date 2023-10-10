import numpy as np
from ReadData import ReadData

DataInputFile, DataOutputFile = ReadData("Data.dat")

# Donnees d'entrainement
DataInput = np.array(DataInputFile)
DataOutput = np.array(DataOutputFile)

input_dim = DataInput.shape[1] # Nombre de neurone pour l'input
hidden_dim1 = 16 # Nombre de neurone de la couche cachee 1 ; 256 
hidden_dim2 = 16 # Nombre de neurone de la couche cachee 1 ; 128
hidden_dim3 = 16 # Nombre de neurone de la couche cachee 1 ; 64
output_dim = 1 # Nombre de neurone pour l'output

# Taux d'apprentissage
learning_rate = 0.01
learning = 100000

# ---------------------


weights_hidden1 = np.random.randn(input_dim, hidden_dim1) # Poids et biais de la couche cachee 1
bias_hidden1 = np.zeros((1, hidden_dim1))

weights_hidden2 = np.random.randn(hidden_dim1, hidden_dim2) # Poids et biais de la couche cachee 2
bias_hidden2 = np.zeros((1, hidden_dim2))

weights_hidden3 = np.random.randn(hidden_dim2, hidden_dim3) # Poids et biais de la couche cachee 3
bias_hidden3 = np.zeros((1, hidden_dim3))

weights_output = np.random.randn(hidden_dim3, output_dim) # Poids et biais de la couche output
bias_output = np.zeros((1, output_dim))

# -------------

# Fonction d'activation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivee de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Modifiez la fonction d'activation pour la couche de sortie
def linear_activation(x):
    return x

# Entrainement du modele
for iteration in range(learning):
    # Propagation
    hidden_layer_activation1 = np.dot(DataInput, weights_hidden1) + bias_hidden1 # Couche cache 1
    hidden_layer_output1 = sigmoid(hidden_layer_activation1)

    hidden_layer_activation2 = np.dot(hidden_layer_output1, weights_hidden2) + bias_hidden2 # Couche cache 2
    hidden_layer_output2 = sigmoid(hidden_layer_activation2)

    hidden_layer_activation3 = np.dot(hidden_layer_output2, weights_hidden3) + bias_hidden3 # Couche cache 3
    hidden_layer_output3 = sigmoid(hidden_layer_activation3)
    
    output_layer_activation = np.dot(hidden_layer_output3, weights_output) + bias_output # Couche sortie
    output_layer_output = sigmoid(output_layer_activation)
    
    # Calcul de l'erreur
    error = np.mean(np.square(DataOutput - output_layer_output))
    
    # Retropropagation 
    d_output = error * sigmoid_derivative(output_layer_output) # Calcul des gradients pour la couche output

    error_hidden_layer3 = d_output.dot(weights_output.T) # Retropropagation pour la couche 3 cachee
    d_hidden_layer3 = error_hidden_layer3 * sigmoid_derivative(hidden_layer_output3)
    error_hidden_layer2 = d_hidden_layer3.dot(weights_hidden3.T) # Retropropagation pour la couche 2 cachee
    d_hidden_layer2 = error_hidden_layer2 * sigmoid_derivative(hidden_layer_output2)
    error_hidden_layer1 = d_hidden_layer2.dot(weights_hidden2.T) # Retropropagation pour la couche 1 cachee
    d_hidden_layer1 = error_hidden_layer1 * sigmoid_derivative(hidden_layer_output1)
    
    # Mise a jour des poids et biais
    weights_hidden3 += hidden_layer_output2.T.dot(d_hidden_layer3) * learning_rate # Mise a jour des poids et biais pour la couche 3 cachee
    bias_hidden3 += np.sum(d_hidden_layer3, axis=0, keepdims=True) * learning_rate
    weights_hidden2 += hidden_layer_output1.T.dot(d_hidden_layer2) * learning_rate # Mise a jour des poids et biais pour la couche 2 cachee
    bias_hidden2 += np.sum(d_hidden_layer2, axis=0, keepdims=True) * learning_rate
    weights_hidden1 += DataInput.T.dot(d_hidden_layer1) * learning_rate # Mise a jour des poids et biais pour la couche 1 cachee
    bias_hidden1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * learning_rate

    weights_output += hidden_layer_output3.T.dot(d_output) * learning_rate # Mise a jour des poids et biais pour la couche output
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate


# Prediction
def PredictingResults(value):
    value_np = np.array(value)
    hidden_layer_activation1 = np.dot(value_np, weights_hidden1) + bias_hidden1 # Couche cacher 1
    hidden_layer_output1 = sigmoid(hidden_layer_activation1)

    hidden_layer_activation2 = np.dot(hidden_layer_output1, weights_hidden2) + bias_hidden2 # Couche cacher 2
    hidden_layer_output2 = sigmoid(hidden_layer_activation2)

    hidden_layer_activation3 = np.dot(hidden_layer_output2, weights_hidden3) + bias_hidden3 # Couche cacher 3
    hidden_layer_output3 = sigmoid(hidden_layer_activation3)

    output_layer_activation = np.dot(hidden_layer_output3, weights_output) + bias_output
    predicted_output = linear_activation(output_layer_activation)  # Utilisez la fonction d'activation linéaire

    return predicted_output


print("Sortie predite apres apprentissage :")
print(PredictingResults([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
