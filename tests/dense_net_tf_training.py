import sys
sys.path.insert(0, ".")
import numpy as np
from MLExp.core.tf_applications.neural_net_classes import DenseNetwork

if __name__ == "__main__":

    data_path = 'MLExp/data/'
    #case = 'Oscillator'
    case = 'Lorenz'

    variables_file = data_path + case + '_variables.npy'
    derivatives_file = data_path + case + '_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0]/2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    model_name = case + "_tf_surrogate"

    input_dim = input_cube.shape[1]
    output_dim = output_cube.shape[1]

    test_setup = {
                  'layers_cells_list': [input_dim, 50, 50, output_dim],
                  'dropouts_rates_list': [0, 0],
                  'learning_rate': 1e-05,
                  'l2_reg': 1e-05,
                  'activation_function': 'elu',
                  'loss_function': 'mse',
                  'optimizer': 'adam',
                  'n_epochs': 20000,
                  'outputpath': data_path,
                  'model_name': model_name,
                  'input_dim': input_dim,
                  'output_dim': output_dim
                  }


    neural_net = DenseNetwork(test_setup)


    neural_net.fit(input_cube, output_cube)

    print("Model constructed.")

