import numpy as np

from networks.network import Network

class DeepQLearning:
    """Deep Q Learning algorithm and all related components"""
    # TODO: refactor out epsilon
    # TODO: consider tradeoffs between configuration vs convenience in setting
    #       the input and output size of the network
    def __init__(self, gamma, epsilon, num_actions, observation_dim,
            learning_rate, buffer_capacity):
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.buffer_capacity = buffer_capacity

        self.network = self.__build_network(learning_rate)

    def act(self, env):
        # TODO: implement replay buffer
        # Perform action from network
        current_state = env.observation
        current_action = self.__get_action(current_state)
        env.step(current_action)
        next_state = env.observation
        reward = env.reward

        self.__update_network(current_state, current_action, reward, next_state,
                env.is_complete())

    def __update_network(self, current_state, current_action, reward, next_state, is_complete):
        """Update network based on states, actions, and rewards"""
        target = self.__calculate_target(is_complete, reward, next_state)

        # Take the entire action vector of the "current_state" and update the
        # action that we took to the target reward. This way, all actions that
        # we didn't take will have an error of 0 when we subtract out the Q
        # estimation.
        current_state_vector = self.network.predict(current_state)
        current_state_vector[0][current_action] = target
        self.network.update(current_state, current_state_vector)

    def __calculate_target(self, is_complete, reward, next_state):
        if is_complete:
            return reward
        else:
            return reward + self.gamma * self.network.max_output(next_state)


    def __get_action(self, input_observation):
        if self.__should_explore():
            return self.__random_action()
        else:
            return self.network.best_action(input_observation)

    def __should_explore(self):
        return np.random.rand() <= self.epsilon

    def __build_network(self, learning_rate):
        input_node_count = np.prod(self.observation_dim)
        output_node_count = self.num_actions
        return Network(input_node_count, output_node_count, learning_rate)

    def __random_action(self):
        return np.random.randint(self.num_actions)
