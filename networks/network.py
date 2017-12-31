from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam

import numpy as np

class Network:
    def __init__(self, input_dimension, output_dimension, learning_rate):
        self.model = self.__build_model(input_dimension, output_dimension, learning_rate)

    def max_output(self, input_state):
        """Returns the highest output value of the network"""
        action_values = self.predict(input_state)
        return np.amax(action_values)

    def best_action(self, input_state):
        """Returns the index of the action with the highest value"""
        action_values = self.predict(input_state)
        return np.argmax(action_values)

    def predict(self, input_state):
        """Predicts the Q values for each action given the state"""
        input_as_matrix = np.reshape(input_state, (-1, input_state.size))
        return np.squeeze(self.model.predict(input_as_matrix, verbose=0))

    def update(self, input_states, labels):
        """Update the weights based on the input_states and the labels

        This updates the network based on the given input data and the
        corresponding labels. While the parameters are plural, this will be
        able to handle a single example or multiple examples.

        Parameters
        ----------
        input_states : numpy array
            Training data for the network. Must match the dimensions of the
            input for the model.
        labels : numpy array
            Label data for the network. Must match the dimensions of the output
            for the model.
        """
        self.model.fit(input_states, targets, verbose=0)

    def __build_model(self, input_dimension, output_dimension, learning_rate):
        model = Sequential()
        # First layer
        model.add(Dense(24, input_dim = input_dimension, activation='relu'))
        # Hidden layer
        model.add(Dense(24, activation='relu'))
        # Output layer
        model.add(Dense(output_dimension, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr = learning_rate))
        return model
