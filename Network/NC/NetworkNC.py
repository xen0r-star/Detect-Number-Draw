import numpy as np
import sys
import time
import os
import signal
import pickle
from ReadData import ReadData
from ProgressBarNC import update_progress_bar

# --------[ Valeur ]--------

learning_rate = 0.01
modele = "Model/model2.pkl"

loss_history = []
iteration = 0
loss = 0

start_time = time.time()
time_update = 0
time_all = 0

# --------[ Donnee ]--------

print("Loading data...")

DataInputFile, DataOutputFile = ReadData("Data.dat")
DataInput = np.array(DataInputFile)
DataOutput = np.array(DataOutputFile)

# Mélenge des donnees
indices = np.arange(len(DataInput))
np.random.shuffle(indices)

DataInput = DataInput[indices]
DataOutput = DataOutput[indices]

if not os.path.exists(modele):
    weights  = np.random.rand(784)
    bias = np.random.rand()
else :
    with open(modele, 'rb') as file:
        params = pickle.load(file)

    weights = params['weights']
    bias = params['bias']
    iteration = params['iteration']
    loss = params['loss']
    time_all = params['time_all']

# --------[ Affichage ]--------

def show_time(temps_total):
    heures = temps_total // 3600
    minutes = (temps_total % 3600) // 60
    secondes = temps_total % 60
    temps_formate = "{:02d}:{:02d}:{:02d}".format(int(heures), int(minutes), int(secondes))
    return temps_formate

print(f"Rate : {learning_rate}\nLearning : {iteration}\nTime : {show_time(time_all)}\nModel : {modele}")
print("\nProgress learning neural networks...")

# --------[ Sauvegarde ]--------

def handle_interrupt(signal, frame):
    print("\n\nInterrupt, save settings...")
    params = {
        'weights': weights,
        'bias': bias,
        'iteration': iteration,
        'loss': loss,
        'time_all': time_all
    }

    with open(modele, 'wb') as file:
        pickle.dump(params, file)
    
    print("Parameters saved, output")
    exit(0)
signal.signal(signal.SIGINT, handle_interrupt)

# --------[ Apprentissage ]--------

while True:
    for i in range(len(DataInput)):
        predicted_output = np.dot(DataInput[i], weights) + bias

        error = DataOutput[i][0] - predicted_output
        loss = np.mean(error**2)
        loss_history.append(loss)

        weights += learning_rate * error * DataInput[i]
        bias += learning_rate * error
    
    iteration += 1

    current_time = time.time() - start_time
    time_all = (current_time - time_update) + time_all
    time_update = current_time

    time_text = show_time(current_time)
    update_progress_bar(iteration, loss, loss_history[len(loss_history) - 2] - loss, time_text)


# test_input = np.array([-52, -51, -50])
# predicted_output = np.dot(test_input, weights) + bias
# print("\nPrediction pour [-52, -51, -50]:", int(predicted_output))