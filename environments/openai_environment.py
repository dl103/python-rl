from environments.base_environment import BaseEnvironment

import gym
import numpy as np

class OpenAIEnvironment(BaseEnvironment):
    def __init__(self, env_identifier, render = False):
        self.env = gym.make(env_identifier)
        self.observation = self.env.reset()
        self.reward = 0
        self.complete = False
        self.render = render

    def step(self, action):
        self.observation, self.reward, self.complete, _ = self.env.step(action)

        if self.render:
            self.env.render()

    def num_actions(self):
        return self.env.action_space.n

    def random_action(self):
        return self.env.action_space.sample()

    @property
    def observation_dimensions(self):
        return self.env.observation_space.shape[0]

    def is_complete(self):
        return self.complete

    def reset(self):
        self.env.reset()
