import abc

class BaseEnvironment(metaclass=abc.ABCMeta):
    """
    Defines an interface for implementing new environments to be used with the
    RL algorithms.
    """

    @abc.abstractmethod
    def step(self, action):
        """
        Steps through one time step of the environment while applying the given
        action
        """
        pass

    @abc.abstractmethod
    def num_actions(self):
        """Returns the number of actions the agent can take in the environment.
        """
        pass

    @abc.abstractmethod
    def random_action(self):
        """Returns the index of a random action
        """
        pass

    @property
    @abc.abstractmethod
    def observation_dimensions(self):
        """Returns a tuple corresponding to the observation's dimensions.
        Returns
        -------
        tuple
            Observation dimensions in the form of (row, col)
        """
        pass

    @abc.abstractmethod
    def is_complete(self):
        """Returns True if the environment's episode is complete.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets the environment back to the initial state.
        """
        pass
