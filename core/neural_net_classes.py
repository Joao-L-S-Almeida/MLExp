import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

class Oscillator(t_max):

    def __init__(self):

        pass

    def __call__(self):

        alpha1 = -0.1
        alpha2 = 2
        beta1 = -2
        beta2 = 0.1

        f = alpha1*(x**2) + beta1*(y**2)
        g = alpha2*(x**2) + beta2*(y**2)

        return np.array([f, g])

class FeedForwardNet:

    def __init__(self):
        pass
    def __call__(self):
        pass

if __name__ == "__main__":

    parser = ArgumentParser(description='Input arguments for the neural network training')
    parser.add_argument(
        '-m', '--model_name', type=str, help='Model name')

    args = parser.parse_args()
