import numpy as np

class Agent:
    def __init__(self, env, network, epsilon):
        print("init agent")
        self.env = env
        self.network = network
        self.epsilon = epsilon

    # Gets the action given the current state. Includes epsilon exploring.
    def get_action(self):
        if np.random.rand() <= self.epsilon:
            print("Act randomly")
            return self.env.action_space().sample()
        else:
            print("Selecting network action")
            return self.network.get_action(self.env.observation)
