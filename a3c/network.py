from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Network:
    def __init__(self, input_dimension, output_dimension, learning_rate):
        print("Initting network")
        self.build_model(input_dimension, output_dimension, learning_rate)

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
