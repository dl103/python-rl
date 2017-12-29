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
    def action_space(self):
        pass

    @abc.abstractmethod
    def observation_space(self):
        pass

    @abc.abstractmethod
    def is_complete(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
