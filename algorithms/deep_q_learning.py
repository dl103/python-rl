import numpy as np

from deep_q.network import Network

class DeepQLearning:
    """Deep Q Learning algorithm and all related components"""
    # Can refactor out epsilon later
    def __init__(self, gamma, epsilon, num_actions, observation_dim):
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.network = self.__build_network()

    def act(self, env):
        # Perform action from network
        current_state = env.observation
        current_action = self.__get_action(current_state)
        env.step(current_action)
        next_state = env.observation
        reward = env.reward

        # Update network/collect rewards
        if env.is_complete():
            target = reward
        else:
            target = reward + self.gamma * self.network.max_output(next_state)

        # Take the entire action vector of the "current_state" and update the
        # action that we took to the target reward. This way, all actions that
        # we didn't take will have an error of 0 when we subtract out the Q
        # estimation.
        current_state_vector = self.network.predict(current_state)
        current_state_vector[0][current_action] = target
        self.network.update(current_state, current_state_vector)

    def __get_action(self, input_observation):
        if np.random.rand() <= self.epsilon:
            return self.__random_action()
        else:
            return self.network.best_action(input_observation)

    def __build_network(self):
        input_node_count = np.prod(self.observation_dim)
        output_node_count = self.num_actions
        return Network(input_node_count, output_node_count, 0.001)

    def __random_action(self):
        return np.random.randint(self.num_actions)
