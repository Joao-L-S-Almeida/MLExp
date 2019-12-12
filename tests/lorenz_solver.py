import numpy as np

from problem_classes import LorenzSystem
from numerics.timeint import RK4
import matplotlib.pyplot as plt

# Testing to solve a nonlinear oscillator problem using
# a 4th order and 4 steps Runge-Kutta

if __name__ == "__main__":

    initial_state = np.array([1, 0, 0])
    T = 50
    dt = 0.001

    rho = 28
    beta = 8/3
    sigma = 10
    problem = LorenzSystem(rho, sigma, beta)

    solver = RK4(problem)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    current_state = initial_state

    for tt in range(time.shape[0]):

        variables_state, derivatives_state = solver.step(current_state, dt)
        variables_timesteps.append(variables_state[:, None])
        derivatives_timesteps.append(derivatives_state[:, None])
        current_state = variables_state

    variables_matrix = np.hstack(variables_timesteps)
    derivatives_matrix = np.hstack(derivatives_timesteps)

    plt.plot(time, variables_matrix[0, :], label="x")
    plt.plot(time, variables_matrix[1, :], label="y")
    plt.plot(time, variables_matrix[2, :], label="z")

    np.save("MLExp/data/Lorenz_variables.npy", variables_matrix)
    np.save("MLExp/data/Lorenz_derivatives.npy", derivatives_matrix)

    plt.xlabel("Time(s)")
    plt.title("Lorenz System")
    plt.legend()

    plt.grid(True)

    plt.show()
