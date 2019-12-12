import numpy as np

class NonlinearOscillator:

    def __init__(self, initial_state, T):

        self.initial_state = initial_state
        self.T = T

        self.alpha1 = -0.1
        self.alpha2 = -2
        self.beta1 = 2
        self.beta2 = -0.1

    def __call__(self, x, y):

        f = self.alpha1*(x**3) + self.beta1*(y**3)
        g = self.alpha2*(x**3) + self.beta2*(y**3)

        return np.array([f, g])