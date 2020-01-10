import sys
sys.path.insert(0, ".")
import numpy as np
from MLExp.core.tf_applications.neural_net_classes import DenseNetwork
from numerics.timeint import RK4, FunctionWrapper
import json

def prediction(neural_net, test_output_cube):

    # Using the derivatives surrogate for time-integrating
    right_operator = FunctionWrapper(neural_net.predict)

    solver = RK4(right_operator)

    initial_state = test_input_cube[-1, :]

    time = 0
    T_max = 25
    dt = 0.001
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

    estimated_variables = np.vstack(estimated_variables)

    print("Extrapolation concluded.")

    for ss in range(estimated_variables.shape[1]):
        error = np.linalg.norm(estimated_variables[ss, :] - test_output_cube[ss, :], 2)
        relative_error = 100 * error / np.linalg.norm(test_input_cube, 2)

        print("Variable series {}, L2 error evaluation: {}".format(ss, relative_error))

    return relative_error

def exec_setups(setups, input_dim, output_dim, test_input_cube):

    errors_dict = dict()

    for setup_key, test_setup in setups.items():

        model_name = "Oscillator_tf_surrogate" + '_' + setup_key

        test_setup['outputpath'] = data_path
        test_setup['model_name'] = model_name
        test_setup['input_dim'] = input_dim
        test_setup['output_dim'] = output_dim

        neural_net = DenseNetwork(test_setup)

        neural_net.fit(input_cube, output_cube)

        relative_error = prediction(neural_net, test_input_cube)
        errors_dict[setup_key] = relative_error
        print("Model constructed.")

    return errors_dict

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

    test_input_cube = variables[training_dim:, :]
    test_output_cube = derivatives[training_dim:, :]

    input_dim = input_cube.shape[1]
    output_dim = output_cube.shape[1]

    fp = open(setups_file, "r")
    setups = json.load(fp)

    error_dict = exec_setups(setups, input_dim, output_dim, test_output_cube)
    print("Execution concluded.")
