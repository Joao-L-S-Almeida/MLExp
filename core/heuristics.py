import numpy as np
import random

class TabuSearch:

    def __init__(self, config):

        self.n_disturbances = config['n_disturbances']
        self.disturbance_list = config['disturbance_list']
        self.tabu_list = list()

    def generate(self, origin_setup):

        modified_setup = origin_setup

        for ii, (field, field_value) in enumerate(origin_setup.items()):

            if isinstance(field_value, list):

                lenght = len(field_value)
                disturbance = self.disturbance_list[ii]
                interval = np.arange(-disturbance, disturbance, 1)
                disturbances = [random.choice(interval) for jj in range(lenght)]
                disturbed = np.array(origin_setup) + np.array(disturbance)
                disturbed = disturbed.tolist()

                modified_setup[field] = disturbed

            else:

                disturbance = self.disturbance_list[ii]
                interval = np.arange(-disturbance, disturbance, 1)
                disturbance = random.choice(interval)
                disturbed = np.array(origin_setup) + np.array(disturbance)
                disturbed = disturbed.tolist()

                modified_setup[field] = disturbed

    def __call__(self, origin_setup_0):

        self.tabu_list.append(origin_setup_0)
        origin_setup = origin_setup_0

        new_setups = list()

        while len(new_setups) < self.n_disturbances:

            new_setup = self.generate(origin_setup)

            if isinstance(new_setup, list):
                new_setups.append(new_setup)

        return new_setups





