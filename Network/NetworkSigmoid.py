import numpy as np
import signal
import pickle
import os
import time
from ReadData import ReadData
from ProgressBar import *

DataInputFile, DataOutputFile = ReadData("Data.dat")

# Donnees d'entrainement
DataInput = np.array(DataInputFile)
DataOutput = np.array(DataOutputFile)

modele = "modele1.pkl"

input_dim = DataInput.shape[1] # Nombre de neurone pour l'input
hidden_dim1 = 256 # Nombre de neurone de la couche cachee 1 ; 256 
hidden_dim2 = 256 # Nombre de neurone de la couche cachee 1 ; 128
hidden_dim3 = 128 # Nombre de neurone de la couche cachee 1 ; 64
output_dim = 1 # Nombre de neurone pour l'output

# Taux d'apprentissage
learning_rate = 0.0001

# Historique des pertes
loss_history = []
iteration = 0
loss = 0

# Temps
start_time = time.time()
time_update = 0

# -------------

# Donner modele
if not os.path.exists(modele):
    weights_hidden1 = np.random.randn(input_dim, hidden_dim1) # Poids et biais de la couche cachee 1
    bias_hidden1 = np.zeros((1, hidden_dim1))

    weights_hidden2 = np.random.randn(hidden_dim1, hidden_dim2) # Poids et biais de la couche cachee 2
    bias_hidden2 = np.zeros((1, hidden_dim2))

    weights_hidden3 = np.random.randn(hidden_dim2, hidden_dim3) # Poids et biais de la couche cachee 3
    bias_hidden3 = np.zeros((1, hidden_dim3))

    weights_output = np.random.randn(hidden_dim3, output_dim) # Poids et biais de la couche output
    bias_output = np.zeros((1, output_dim))

    # Autre donnee
    time_all = 0
else :
    with open(modele, 'rb') as file:
        params = pickle.load(file)

    # Recuperation des poids et biais
    weights_hidden1 = params['weights_hidden1']
    bias_hidden1 = params['bias_hidden1']
    weights_hidden2 = params['weights_hidden2']
    bias_hidden2 = params['bias_hidden2']
    weights_hidden3 = params['weights_hidden3']
    bias_hidden3 = params['bias_hidden3']
    weights_output = params['weights_output']
    bias_output = params['bias_output']
    iteration = params['iteration']
    loss = params['loss']
    time_all = params['time_all']

# -------------

#Temps
def show_time(temps_total):
    heures = temps_total // 3600
    minutes = (temps_total % 3600) // 60
    secondes = temps_total % 60
    temps_formate = "{:02d}:{:02d}:{:02d}".format(int(heures), int(minutes), int(secondes))
    return temps_formate

# Stat
time_text = show_time(time_all)
print(f"+---------+------------+----------+\n|  \033[91mLayer\033[0m  |  \033[91mNumber N\033[0m  |   \033[91mLink\033[0m   |\n+---------+------------+----------+\n| \033[94mInput\033[0m   | \033[92m{str(input_dim) + ' ' * (10 - len(str(input_dim)))}\033[0m | \033[93m-\033[0m        |\n| \033[96mHidden1\033[0m | \033[92m{str(hidden_dim1) + ' ' * (10 - len(str(hidden_dim1)))}\033[0m | \033[93m{str(input_dim * hidden_dim1) + ' ' * (8 - len(str(input_dim * hidden_dim1)))}\033[0m |\n| \033[96mHidden2\033[0m | \033[92m{str(hidden_dim2) + ' ' * (10 - len(str(hidden_dim2)))}\033[0m | \033[93m{str(hidden_dim1 * hidden_dim2) + ' ' * (8 - len(str(hidden_dim1 * hidden_dim2)))}\033[0m |\n| \033[96mHidden3\033[0m | \033[92m{str(hidden_dim3) + ' ' * (10 - len(str(hidden_dim3)))}\033[0m | \033[93m{str(hidden_dim2 * hidden_dim3) + ' ' * (8 - len(str(hidden_dim2 * hidden_dim3)))}\033[0m |\n| \033[94mOutput\033[0m  | \033[92m{str(output_dim) + ' ' * (10 - len(str(output_dim)))}\033[0m | \033[93m{str(hidden_dim3 * output_dim) + ' ' * (8 - len(str(hidden_dim3 * output_dim)))}\033[0m |\n+---------+------------+----------+\n\033[91mRate\033[0m : \033[96m{learning_rate}\033[0m\n\033[91mLearning\033[0m : \033[96m{iteration}\033[0m\n\033[91mTime\033[0m : \033[96m{time_text}\033[0m\n\033[91mModel\033[0m : \033[96m{modele}\033[0m")
print("\n\033[92mProgress learning neural networks...\033[0m")

# -------------

# Fonction d'interruption et de sauvegarde
def handle_interrupt(signal, frame):
    print("\n\n\033[91mInterrupt, save settings...\033[0m")
    params = {
        'weights_hidden1': weights_hidden1,
        'bias_hidden1': bias_hidden1,
        'weights_hidden2': weights_hidden2,
        'bias_hidden2': bias_hidden2,
        'weights_hidden3': weights_hidden3,
        'bias_hidden3': bias_hidden3,
        'weights_output': weights_output,
        'bias_output': bias_output,
        'iteration': iteration,
        'loss': loss,
        'time_all': time_all
    }

    with open(modele, 'wb') as file:
        pickle.dump(params, file)
    
    print("\033[92mParameters saved, output\033[0m")
    exit(0)
signal.signal(signal.SIGINT, handle_interrupt)

# Fonction d'activation (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivee de la fonction d'activation sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Entrainement du modele
while True:
    iteration += 1

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
    error = DataOutput - output_layer_output

    # Calcul de la perte
    loss = np.mean(error**2)
    loss_history.append(loss)

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


    current_time = time.time() - start_time
    time_all = (current_time - time_update) + time_all
    time_update = current_time

    time_text = show_time(current_time)
    update_progress_bar(iteration, loss, loss_history[len(loss_history) - 2] - loss, time_text)


# Prediction
# def PredictingResults(value):
#     hidden_layer_activation1 = np.dot(value, weights_hidden1) + bias_hidden1 # Couche cacher 1
#     hidden_layer_output1 = sigmoid(hidden_layer_activation1)

#     hidden_layer_activation2 = np.dot(hidden_layer_output1, weights_hidden2) + bias_hidden2 # Couche cacher 2
#     hidden_layer_output2 = sigmoid(hidden_layer_activation2)

#     hidden_layer_activation3 = np.dot(hidden_layer_output2, weights_hidden3) + bias_hidden3 # Couche cacher 3
#     hidden_layer_output3 = sigmoid(hidden_layer_activation3)

#     output_layer_activation = np.dot(hidden_layer_output3, weights_output) + bias_output
#     predicted_output = output_layer_activation # Utilisez la fonction d'activation linéaire

#     return predicted_output


# print("\n\nSortie predite apres apprentissage :")
# print(PredictingResults([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
# # Reponse attendue 4