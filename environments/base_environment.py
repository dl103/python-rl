import abc

STEP_MUL = 16

class BaseEnvironment(metaclass=abc.ABCMeta):
    """
    Defines an interface for implementing new environments to be used with the
    RL algorithms.
    """

    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def num_actions(self):
        pass

    @abc.abstractmethod
    def random_action(self):
        pass

    @property
    @abc.abstractmethod
    def observation_dimensions(self):
        """
        Returns a tuple of (row, col) corresponding to the observations dimensions.
        """
        pass

    @abc.abstractmethod
    def is_complete(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
