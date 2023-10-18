import numpy as np
import sys
import signal

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

loss_history = []

# Poids et biais de la couche cachee
weights_hidden = np.random.uniform(size=(input_dim, hidden_dim))
bias_hidden = np.random.uniform(size=(1, hidden_dim))

# Poids et biais de la couche de sortie
weights_output = np.random.uniform(size=(hidden_dim, output_dim))
bias_output = np.random.uniform(size=(1, output_dim))

print(f"\033[91mNeuron\033[0m : \033[96m{hidden_dim}\033[91m\nRate\033[0m : \033[96m{learning_rate}\n\033[91mLearning\033[0m : \033[96m{learning}")
print("\n\033[92mProgress learning neural networks...\033[0m")

def update_progress_bar(iteration, total, loss, variation):
    length=50
    progress = str(iteration).zfill(len(str(total)))
    percent = int(iteration / total * 100)
    progress_bar = '━' * int(length * iteration / total) + \
                   ' ' * (length - int(length * iteration / total))
    
    if percent >= 85:
        color1 = '\033[92m'
    elif percent >= 50:
        color1 = '\033[93m'
    else:
        color1 = '\033[91m'

    if loss <= 1:
        color2 = '\033[92m'
    elif loss <= 10:
        color2 = '\033[93m'
    else:
        color2 = '\033[91m'

    sys.stdout.write('\r\033[96m{}/{}\033[0m {} {}\033[0m% | Loss : {}\033[0m | Variation : \033[93m{:.15f}\033[0m'.format(progress, total, color1 + progress_bar, percent, color2 + str(loss), variation))
    sys.stdout.flush()

# Fonction d'activation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivee de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Entrainement du modele
for iteration in range(learning):
    # Propagation
    hidden_layer_activation = np.dot(DataInput, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_output) + bias_output
    output_layer_output = sigmoid(output_layer_activation)
    
    # Calcul de l'erreur
    error = DataOutput - output_layer_output
    loss = np.mean(error**2)
    loss_history.append(loss)
    
    # Retropropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Mise a jour des poids et biais
    weights_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += DataInput.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    update_progress_bar(iteration + 1, learning, loss, loss_history[len(loss_history) - 2] - loss)

# Prediction
def PredictingResults(value):
    hidden_layer_activation = np.dot(value, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, weights_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)
    return predicted_output

def handle_interrupt(signal, frame):
    print("\033[0m")
    exit(0)
signal.signal(signal.SIGINT, handle_interrupt)

print("\n")
while True:
    a = int(input("Le premier nombre booléen : \033[96m"))
    b = int(input("\033[0mLe deuxième nombre booléen : \033[91m"))
    print(f"\033[0mLa reponse de [\033[96m{a}\033[0m And \033[91m{b}\033[0m] est : \033[93m{round(PredictingResults([a,b])[0][0])}\033[0m \n")
