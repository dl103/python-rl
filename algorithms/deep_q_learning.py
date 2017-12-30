import numpy as np

from networks.network import Network

class DeepQLearning:
    """Deep Q Learning algorithm and all related components"""
    # TODO: refactor out epsilon
    # TODO: consider tradeoffs between configuration vs convenience in setting
    #       the input and output size of the network
    def __init__(self, gamma, epsilon, num_actions, observation_dim, learning_rate):
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.observation_dim = observation_dim

        self.network = self.__build_network(learning_rate)

    def act(self, env):
        # TODO: implement replay buffer
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
        if self.__should_explore():
            return self.__random_action()
        else:
            return self.network.best_action(input_observation)

    def __should_explore(self):
        return np.random.rand() <= self.epsilon

    def __build_network(self):
        input_node_count = np.prod(self.observation_dim)
        output_node_count = self.num_actions
        return Network(input_node_count, output_node_count, self.learning_rate)

    def __random_action(self):
        return np.random.randint(self.num_actions)
