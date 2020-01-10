import sys
sys.path.insert(0, ".")
import numpy as np
from MLExp.core.tf_applications.neural_net_classes import DenseNetwork
import json

if __name__ == "__main__":

    data_path = 'MLExp/data/'

    variables_file = data_path + 'Oscillator_variables.npy'
    derivatives_file = data_path + 'Oscillator_derivatives.npy'
    setups_file = data_path + "setups.json"

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0]/2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    input_dim = input_cube.shape[1]
    output_dim = output_cube.shape[1]

    fp = open(setups_file, "r")
    setups = json.load(fp)

    for setup_key, test_setup in setups.items():

        model_name = "Oscillator_tf_surrogate" + '_' + setup_key

        test_setup['outputpath'] = data_path
        test_setup['model_name'] = model_name
        test_setup['input_dim'] = input_dim
        test_setup['output_dim'] = output_dim

        neural_net = DenseNetwork(test_setup)

        neural_net.fit(input_cube, output_cube)

        print("Model constructed.")

