import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from core.rom import POD
from numerics.timederivation import CollocationDerivative
from core.tf_applications.neural_net_classes import DenseNetwork
from numerics.timeint import RK4, FunctionWrapper

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--problem', type=str)

    args = parser.parse_args()

    data_path = args.data_path
    model_name = args.model_name
    problem = args.problem

    # Manufactured data
    t_min = 0
    t_max = 1

    x_min = 0
    x_max = 1

    y_min = 0
    y_max = 1

    N_t = 500
    N_x = 100
    N_y = 100

    omega_t = 5*np.pi
    lambd = 1

    t = np.linspace(t_min, t_max, N_t)
    x = np.linspace(x_min, x_max, N_x)
    y = np.linspace(y_min, y_max, N_y)

    X, T, Y = np.meshgrid(x, t, y)

    dt = (t_max - t_min) / N_t

    if problem == "time_interact":
        U = np.exp(-lambd*Y*T)*np.cos(omega_t*T*X)
    elif problem == "time_isolated":
        U = (np.exp(-lambd * T)*(T**2))\
            * np.cos(omega_t * Y * X**3)*np.cos(omega_t * Y**2 * X)
    elif problem == "time_oscillating":
        U = (np.cos(omega_t*T) + np.cos(omega_t*T/2) + np.cos(omega_t*T/4)) \
            * np.cos(omega_t * Y * X ** 3) * np.cos(omega_t * Y ** 2 * X)
    else:
        raise Exception("Problem no implemented.")

    # Preparing data to be used in the ROM
    shapes = U.shape
    collapsible_shapes = shapes[1:]
    immutable_shape = shapes[0]
    collapsed_shape = np.prod(collapsible_shapes)

    U_flatten = U.reshape((immutable_shape, collapsed_shape))

    data = U_flatten

    batch_size = data.shape[0]

    train_batch_size = int(batch_size/2)

    T_max = (t_max - t_min)/2

    # Separating training and testing data
    train_data = data[:train_batch_size, :]
    test_data = data[train_batch_size:, :]

    # Instantiating the Proper Orthogonal Decomposition class
    config = {'n_components': 5}
    rom = POD(config=config)

    # Subtracting the mean component
    train_data_mean = train_data.mean(0)[None, :]
    global_norm = np.linalg.norm(train_data, 2)
    mean_norm = np.linalg.norm(train_data_mean, 2)

    train_data = train_data - train_data_mean
    test_data = test_data - train_data_mean

    print("Relative contribution of the mean component: {}".format(mean_norm/global_norm))

    rom.fit(data=train_data)

    reduced_data = rom.project(data=train_data)
    train_data_reconstructed = rom.reconstruct(data=reduced_data)

    projection_error = np.linalg.norm(train_data - train_data_reconstructed, 2)
    ref_value = np.linalg.norm(train_data, 2)

    print("Relative projection error: {} % ".format(100*projection_error/ref_value))

    test_reduced_data = rom.project(data=test_data)

    derivative_op = CollocationDerivative(timestep=dt)

    derivative_reduced_data = derivative_op.solve(data=reduced_data)
    test_derivative_reduced_data = derivative_op.solve(data=test_reduced_data)

    input_data = reduced_data
    output_data = derivative_reduced_data

    input_dim = input_data.shape[1]
    output_dim = output_data.shape[1]

    # This test setup (or multiple setups) can be stored in JSON files and
    # read at run time
    test_setup = {
        'layers_cells_list': [input_dim, 100, 100, 100, 100, 100, output_dim],
        'dropouts_rates_list': [0, 0, 0, 0, 0],
        'learning_rate': 1e-05,
        'l2_reg': 1e-05,  # 1e-05,
        'activation_function': ['elu', 'relu', 'elu', 'relu', 'tanh', 'relu'],
        'loss_function': 'mse_normed',
        'optimizer': 'adam',
        'n_epochs': 50000,
        'outputpath': data_path,
        'model_name': model_name,
        'input_dim': input_dim,
        'output_dim': output_dim
    }

    # It constructs the neural net
    neural_net = DenseNetwork(test_setup)
    # It executes the training process and saves the model
    # in data_path
    neural_net.fit(input_data, output_data)

    estimated_derivative_data_reduced = neural_net.predict(test_reduced_data)

    error = np.linalg.norm(test_derivative_reduced_data
                           - estimated_derivative_data_reduced, 2)

    relative_error = error/np.linalg.norm(test_derivative_reduced_data, 2)

    print("Relative derivative error: {} %".format(100*relative_error))

    ref_value = np.linalg.norm(test_data, 2)

    # Using the derivatives surrogate for time-integrating
    right_operator = FunctionWrapper(neural_net.predict)

    solver = RK4(right_operator)

    initial_state = input_data[-1, :]

    time = 0
    estimated_variables = list()

    N_steps = int(T_max / dt)
    n_steps = test_data.shape[0]
    interval = int(N_steps / n_steps)

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

    estimated_test_data = rom.reconstruct(data=estimated_variables)

    error = np.linalg.norm(test_data - estimated_test_data, 2)

    print("Extrapolation error : {} %".format(100*error/ref_value))

    print("Extrapolation concluded.")

    # Post-processing
    final_shape = collapsible_shapes

    estimated_slice = estimated_test_data[-1, :].reshape(final_shape)
    exact_slice = test_data[-1, :].reshape(final_shape)

    plt.imshow(estimated_slice)
    plt.colorbar()
    plt.show()
    plt.savefig(data_path + "estimated_solution.png")

    plt.imshow(exact_slice)
    plt.colorbar()
    plt.show()
    plt.savefig(data_path + "exact_solution.png")

