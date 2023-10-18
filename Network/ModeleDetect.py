import numpy as np
import pickle

def ModeleDetect(Value, modele) :
    with open(modele, 'rb') as file:
        params = pickle.load(file)

    weights = params['weights']
    bias = params['bias']

    test_input = np.array(Value)
    predicted_output = int(np.dot(test_input, weights) + bias)

    return predicted_output