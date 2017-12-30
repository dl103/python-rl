class Agent:
    def __init__(self, env, network, epsilon):
        self.env = env
        self.network = network
        self.epsilon = epsilon
