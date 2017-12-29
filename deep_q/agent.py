import numpy as np

class Agent:
    def __init__(self, env, network, epsilon):
        self.env = env
        self.network = network
        self.epsilon = epsilon

    # Gets the action given the current state. Includes epsilon exploring.
    def get_action(self):
        if np.random.rand() <= self.epsilon:
            return self.env.random_action()
        else:
            return self.network.best_action(self.env.observation)
