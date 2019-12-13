import numpy as np
from core.neural_net_classes import DenseNetwork

if __name__ == "__main__":

    data_path = 'MLExp/data/'

    variables_file = data_path + 'Oscillator_variables.npy'
    derivatives_file = data_path + 'Oscillator_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    input_cube = variables
    output_cube = derivatives
    model_name = "Oscillator_surrogate.h5"

    test_setup = {
                  'layers_cells_list': [100, 100],
                  'dropouts_rates_list': [0, 0],
                  'learning_rate': 1e-05,
                  'l2_reg' : 1e-06,
                  'activation_function': 'elu',
                  'loss_function': 'mse',
                  'optimizer': 'adam',
                  'n_epochs' : 10000
                  }

    neural_net = DenseNetwork(test_setup)
    neural_net.fit(input_cube, output_cube)
    neural_net.save(data_path+model_name)

    print("Data loaded.")
