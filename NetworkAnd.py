import numpy as np

# Donnees d'entrainement
DataInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
DataOutput = np.array([[0], [0], [0], [1]])

# Initialisation des poids et biais de la couche cachee et de sortie
input_dim = DataInput.shape[1]
hidden_dim = 4
output_dim = 1

# Taux d'apprentissage
learning_rate = 0.1
learning = 10000

# Poids et biais de la couche cachee
weights_hidden = np.random.uniform(size=(input_dim, hidden_dim))
bias_hidden = np.random.uniform(size=(1, hidden_dim))

# Poids et biais de la couche de sortie
weights_output = np.random.uniform(size=(hidden_dim, output_dim))
bias_output = np.random.uniform(size=(1, output_dim))

# Fonction d'activation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivee de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Entrainement du modele
for learning in range(learning):
    # Propagation
    hidden_layer_activation = np.dot(DataInput, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_output) + bias_output
    output_layer_output = sigmoid(output_layer_activation)
    
    # Calcul de l'erreur
    error = DataOutput - output_layer_output
    
    # Retropropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Mise a jour des poids et biais
    weights_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += DataInput.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Prediction
def PredictingResults(value):
    hidden_layer_activation = np.dot(value, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)
    return predicted_output

print("Sortie predite apres apprentissage :")
print(round(PredictingResults([1,0])[0][0]))
print(round(PredictingResults([0,0])[0][0]))
print(round(PredictingResults([0,1])[0][0]))
print(round(PredictingResults([1,1])[0][0]))
