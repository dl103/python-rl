#  from pysc2.env import environment
from pysc2.env import sc2_env
import numpy as np
import pdb

# OpenAI gym
import gym

STEP_MUL = 16

class Environment:
    def __init__(self):
        #  env = sc2_env.SC2Env(
                #  map_name="DefeatZerglingsAndBanelings",
                #  step_mul=STEP_MUL,
                #  visualize=True
            #  )
        self.env = gym.make('CartPole-v0')
        self.observation = np.reshape(self.env.reset(), [1,4])
        self.reward = 0
        self.complete = False
        self.info = None
        print("Init environment")

    def step(self, action):
        self.observation, self.reward, self.complete, self.info = self.env.step(action)
        self.observation = np.reshape(self.observation, [1,4])
        self.env.render()

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space

    def is_complete(self):
        return self.complete

    def reset(self):
        self.env.reset()
