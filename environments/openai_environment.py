from environments.base_environment import BaseEnvironment

import gym
import numpy as np

class OpenAIEnvironment(BaseEnvironment):
    def __init__(self, render = False):
        self.env = gym.make('CartPole-v0')
        self.observation = np.reshape(self.env.reset(), [1,4])
        self.reward = 0
        self.complete = False
        self.info = None
        self.render = render

    def step(self, action):
        self.observation, self.reward, self.complete, self.info = self.env.step(action)
        self.observation = np.reshape(self.observation, [1,4])
        self.env.render() if self.render:

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space

    def is_complete(self):
        return self.complete

    def reset(self):
        self.env.reset()
