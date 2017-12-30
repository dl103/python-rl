import numpy as np

class DeepQLearning:
    """Deep Q Learning algorithm and all related components"""
    # Can refactor out epsilon later
    def __init__(self, gamma, epsilon, num_actions):
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.network = _build_network(num_actions)

    # Let's see if we can do this without passing in the env.
    def get_action(input_observation):
        if np.random.rand() <= self.epsilon:
            return self.env.random_action()
        else:
            return self.network.best_action(self.env.observation)

    def _build_network(self):
        input_node_count = np.prod(env.observation_dimensions)
        output_node_count = self.num_actions
        Network(input_node_count, output_node_count, 0.001)
