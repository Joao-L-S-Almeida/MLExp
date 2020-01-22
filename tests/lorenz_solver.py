import sys
sys.path.insert(0, '.')

import numpy as np
from MLExp.tests.problem_classes import LorenzSystem
from MLExp.numerics.timeint import RK4
import matplotlib.pyplot as plt

from argparse import ArgumentParser


# Testing to solve a nonlinear oscillator problem using
# a 4th order and 4 steps Runge-Kutta

if __name__ == "__main__":

    parser = ArgumentParser(description="Reading input arguments")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--dt', type=float)

    args = parser.parse_args()

    data_path = args.data_path
    T = args.time
    dt = args.dt

    initial_state = np.array([1, 0, 0])

    rho = 28
    beta = 8/3
    sigma = 10
    problem = LorenzSystem(rho, sigma, beta)

    solver = RK4(problem)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    current_state = initial_state

    iter = 0
    for tt in range(time.shape[0]):

        variables_state, derivatives_state = solver.step(current_state, dt)
        variables_timesteps.append(variables_state[:, None])
        derivatives_timesteps.append(derivatives_state[:, None])
        current_state = variables_state

        sys.stdout.write("\rIteration {}".format(iter))
        sys.stdout.flush()
        iter += 1

    variables_matrix = np.hstack(variables_timesteps)
    derivatives_matrix = np.hstack(derivatives_timesteps)

    plt.plot(time, variables_matrix[0, :], label="x")
    plt.plot(time, variables_matrix[1, :], label="y")
    plt.plot(time, variables_matrix[2, :], label="z")

    label_string = "rho_{}_sigma_{}_beta_{}_T_{}_dt_{}".format(rho, sigma, beta, T, dt)

    np.save(data_path + "Lorenz_variables_{}.npy".format(label_string), variables_matrix)
    np.save(data_path + "Lorenz_derivatives_{}.npy".format(label_string), derivatives_matrix)

    plt.xlabel("Time(s)")
    plt.title("Lorenz System")
    plt.legend()

    plt.grid(True)

    plt.show()
