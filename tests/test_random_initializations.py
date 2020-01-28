import sys
sys.path.insert(0, ".")
import numpy as np
from MLExp.core.tf_applications.neural_net_classes import DenseNetwork
from MLExp.numerics.timeint import RK4, FunctionWrapper

from argparse import ArgumentParser

def prediction(neural_net, test_input_cube, choices, initial_state):

    # Using the derivatives surrogate for time-integrating
    right_operator = FunctionWrapper(neural_net.predict)

    solver = RK4(right_operator)

    time = choices['time']
    T_max = choices['T_max']
    dt = choices['dt']
    estimated_variables = list()

    ii = 0
    # Approach based on Lui & Wolf (https://arxiv.org/abs/1903.05206)
    while time < T_max:
        state, derivative_state = solver.step(initial_state, dt)
        estimated_variables.append(state)
        initial_state = state
        sys.stdout.write("\rIteration {}".format(ii))
        sys.stdout.flush()
        time += dt
        ii += 1

    estimated_variables = np.vstack(estimated_variables)[:-1, :]

    print("Extrapolation concluded.")

    for ss in range(estimated_variables.shape[1]):

        error = np.linalg.norm(estimated_variables[:, ss] - test_input_cube[:, ss], 2)
        relative_error = 100 * error / np.linalg.norm(test_input_cube[:, ss], 2)
        log_string = "Variable series {}, L2 error evaluation: {}".format(ss, relative_error)
        print(log_string)


    return relative_error

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--case', type=str)

    args = parser.parse_args()

    data_path = args.data_path
    case = args.case

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

    test_input_cube = variables[training_dim:, :]
    test_output_cube = derivatives[training_dim:, :]

    test_setup = {
                  'layers_cells_list': [input_dim, 50, 50, 50, output_dim],
                  'dropouts_rates_list': [0, 0, 0],
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
    number_of_tests = 10

    initial_state = input_cube[-1, :]

    T = 25
    dt = 0.005

    choices = {
                'time': 0,
                'T_max': T,
                'dt': dt
              }

    errors = list()
    for ss in range(number_of_tests):

        neural_net.fit(input_cube, output_cube)
        error = prediction(neural_net, test_input_cube, choices, initial_state)
        errors.append(error)

    errors = np.array(error)

    print("Model constructed.")

