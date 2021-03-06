import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

class CollocationDerivative:

    def __init__(self, timestep=None):

        self.dt = timestep

    def solve(self, data=None, x=None):

        # data must be a matrix with shape (n_timsteps, n_variables)
        n_variables = data.shape[1]
        n_timesteps = data.shape[0]

        if self.dt:
            times = np.arange(0, n_timesteps, 1)*self.dt

        elif (not self.dt) and isinstance(x, np.ndarray):
            times = x

        else:
            raise Exception('There is no axis to perform differentiation.')

        data_derivatives_list = list()
        for var in range(n_variables):

            var_array = data[:, var]

            interpolator = ius(times, var_array)
            derivative_intp = interpolator.derivative()
            derivatives = derivative_intp(times)

            data_derivatives_list.append(derivatives[:, None])

        data_derivatives = np.hstack(data_derivatives_list)

        return data_derivatives


