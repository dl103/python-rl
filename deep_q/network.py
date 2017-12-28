from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

class Network:
    def __init__(self, input_dimension, output_dimension, learning_rate):
        print("Initting network")
        self.model = self.build_model(input_dimension, output_dimension, learning_rate)

    # TODO: change this architecture to the Deep RL paper architecture
    def build_model(self, input_dimension, output_dimension, learning_rate):
        print("Building model")
        model = Sequential()
        # First layer
        model.add(Dense(24, input_dim = input_dimension, activation='relu'))
        # Hidden layer
        model.add(Dense(24, activation='relu'))
        # Output layer
        model.add(Dense(output_dimension, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr = learning_rate))
        return model

    def get_action(self, input_state):
        action_values = self.model.predict(input_state)
        return np.argmax(action_values[0])
