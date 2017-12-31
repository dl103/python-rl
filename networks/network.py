from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam

import numpy as np
import pdb

class Network:
    def __init__(self, input_dimension, output_dimension, learning_rate):
        self.model = self.build_model(input_dimension, output_dimension, learning_rate)

    # TODO: change this architecture to the Deep RL paper architecture
    def build_model(self, input_dimension, output_dimension, learning_rate):
        model = Sequential()
        # First layer
        model.add(Dense(24, input_dim = input_dimension, activation='relu'))
        # Hidden layer
        model.add(Dense(24, activation='relu'))
        # Output layer
        model.add(Dense(output_dimension, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr = learning_rate))
        return model

    # Returns the highest output value of the network
    def max_output(self, input_state):
        action_values = self.predict(input_state)
        return np.amax(action_values[0])

    # Returns the index of the action with the highest value
    def best_action(self, input_state):
        action_values = self.predict(input_state)
        return np.argmax(action_values[0])

    # Predicts the Q values for each action given the state
    def predict(self, input_state):
        input_as_matrix = np.reshape(input_state, (-1, input_state.size))
        return self.model.predict(input_as_matrix, verbose=0)

    # Update the weights based on the input_state and the target vector
    def update(self, input_state, target):
        pdb.set_trace()
        input_as_matrix = np.reshape(input_state, (-1, input_state.shape[0]))
        self.model.fit(input_as_matrix, target, verbose=0)

    def batch_update(self, input_states, targets):
        self.model.train_on_batch(np.array(input_states), np.array(targets))
