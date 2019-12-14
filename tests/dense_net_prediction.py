import sys

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from numerics.timeint import RK4, FunctionWrapper

if __name__ == "__main__":

    data_path = 'MLExp/data/'

    variables_file = data_path + 'Oscillator_variables.npy'
    derivatives_file = data_path + 'Oscillator_derivatives.npy'

    variables = np.load(variables_file)
    derivatives = np.load(derivatives_file)

    variables = variables.T
    derivatives = derivatives.T

    training_dim = int(variables.shape[0] / 2)

    input_cube = variables[:training_dim, :]
    output_cube = derivatives[:training_dim, :]

    test_input_cube = variables[training_dim:, :]
    test_output_cube = derivatives[training_dim:, :]

    model_name = "Oscillator_surrogate.h5"

    model = load_model(data_path + model_name)

    # Post-processing
    print("Estimating the test output using the trained model.")

    estimated_cube = model.predict(test_input_cube)

    for ss in range(estimated_cube.shape[1]):

        error = np.linalg.norm(estimated_cube[ss, :] - test_output_cube[ss, :], 2)
        relative_error = error / np.linalg.norm(test_output_cube, 2)

        print("Derivative series {}, L2 error evaluation: {}".format(ss, relative_error))

        plt.plot(test_output_cube[:, ss], label="Target")
        plt.plot(estimated_cube[:, ss], label="Estimated")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Using the derivatives surrogate for time-integrating
    right_operator = FunctionWrapper(model.predict)

    solver = RK4(right_operator)

    initial_state = test_input_cube[-1, :]

    time = 0
    T_max = 25
    dt = 0.01
    estimated_variables = list()

    ii = 0
    while time < T_max:

        state, derivative_state = solver.step(initial_state, dt)
        estimated_variables.append(state)
        initial_state = state
        sys.stdout.write("\rIteration {}".format(ii))
        sys.stdout.flush()
        time += dt
        ii += 1

    estimated_variables = np.vstack(estimated_variables)

    print("Extrpolation concluded.")

    for ss in range(estimated_variables.shape[1]):

        error = np.linalg.norm(estimated_variables[ss, :] - test_input_cube[ss, :], 2)
        relative_error = error / np.linalg.norm(test_input_cube, 2)

        print("Derivative series {}, L2 error evaluation: {}".format(ss, relative_error))

        plt.plot(test_input_cube[:, ss], label="Target")
        plt.plot(estimated_variables[:, ss], label="Estimated")
        plt.legend()
        plt.grid(True)
        plt.show()

