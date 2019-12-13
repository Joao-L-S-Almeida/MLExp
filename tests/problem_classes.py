import numpy as np

class NonlinearOscillator:

    def __init__(self):

        self.alpha1 = -0.1
        self.alpha2 = -2
        self.beta1 = 2
        self.beta2 = -0.1

    def __call__(self, x, y):

        f = self.alpha1*(x**3) + self.beta1*(y**3)
        g = self.alpha2*(x**3) + self.beta2*(y**3)

        return np.array([f, g])

class LorenzSystem:

    def __init__(self, rho, sigma, beta):

        self.rho = rho
        self.beta = beta
        self.sigma = sigma

    def __call__(self, x, y, z):

        f = self.sigma * (y - x)
        g = x*(self.rho - z) - y
        h = x*y - self.beta*z

        return np.array([f, g, h])

