import numpy as np

from problem_classes import NonlinearOscillator
from numerics.timeint import RK4
import matplotlib.pyplot as plt

if __name__ == "__main__":

    initial_state = np.array([2,0])
    T = 50
    dt = 0.1
    oscillator = NonlinearOscillator(initial_state, T)
    integrator = RK4(oscillator)

    time = np.arange(0, T, dt)

    variables_timesteps = list()
    derivatives_timesteps = list()

    for tt in range(time.shape[0]):

        variables_state, derivatives_state = integrator.step(initial_state, dt)
        variables_timesteps.append(variables_state[:, None])
        derivatives_timesteps.append(derivatives_state[:, None])
        initial_state = variables_state


    variables_matrix = np.hstack(variables_timesteps)
    derivatives_matrix = np.hstack(derivatives_timesteps)

    plt.plot(time, variables_matrix[0,:])
    plt.plot(time, variables_matrix[1,:])
    plt.show()
