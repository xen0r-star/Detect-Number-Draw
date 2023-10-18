import numpy as np
import sys

DataInput = np.array([[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [4, 5, 6]])
DataOutput = np.array([[5], [6], [7], [8], [7]])

weights  = np.random.rand(3)
bias = np.random.rand()

learning_rate = 0.01
learning = 2500

for iteration in range(learning):
    total_error = 0
    for i in range(len(DataInput)):
        predicted_output = np.dot(DataInput[i], weights) + bias

        error = DataOutput[i][0] - predicted_output
        total_error += np.abs(error)

        weights += learning_rate * error * DataInput[i]
        bias += learning_rate * error

    sys.stdout.write('\rLearning : {} | Loss : {:.15f}'.format(iteration + 1, total_error / len(DataInput)))
    sys.stdout.flush()

print("\n")
while True:
    a = int(input("Le premier nombre de la suite : \033[96m"))
    b = [a, a + 1, a + 2]
    test_input = np.array(b)
    predicted_output = int(np.dot(test_input, weights) + bias)

    if predicted_output != a + 3:
        predicted_output = "Erreur"

    print(f"\033[0mLa prediction pour [\033[96m{b[0]}\033[0m, \033[96m{b[1]}\033[0m, \033[96m{b[2]}\033[0m] est : \033[93m{predicted_output}\033[0m \n")
