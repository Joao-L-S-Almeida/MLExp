import numpy as np
import random

class TabuSearch:

    def __init__(self, config):

        self.n_disturbances = config['n_disturbances']
        self.disturbance_list = config['disturbance_list']
        self.tabu_list = list()

    def generate(self, origin_setup):

        modified_setup = origin_setup

        for field, field_value in self.disturbance_list.items():

            if isinstance(origin_setup[field], list):

                lenght = len(origin_setup[field])
                disturbance = field_value
                interval = np.arange(-disturbance, disturbance+1, 1)
                disturbances = [random.choice(interval) for jj in range(lenght)]
                disturbances[0] = 0
                disturbances[-1] = 0
                disturbed = np.array(origin_setup[field]) + np.array(disturbance)
                disturbed = disturbed.tolist()

                modified_setup[field] = disturbed

            else:

                disturbance = self.disturbance_list[field]
                interval = np.arange(-disturbance, disturbance, 1)
                disturbance = random.choice(interval)
                disturbed = np.array(origin_setup[field]) + np.array(disturbance)
                disturbed = disturbed.tolist()

                modified_setup[field] = disturbed

        return modified_setup

    def __call__(self, origin_setup_0):

        self.tabu_list.append(origin_setup_0)
        origin_setup = origin_setup_0

        new_setups = list()

        while len(new_setups) < self.n_disturbances:

            new_setup = self.generate(origin_setup)

            if isinstance(new_setup, list):
                new_setups.append(new_setup)

        return new_setups





