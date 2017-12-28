#  from pysc2.env import environment
from pysc2.env import sc2_env
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
        self.env.reset()
        print("Init environment")

    def step(self, action):
        self.obs, self.reward, self.complete, self.info = self.env.step(action)
        self.env.render()

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space

    def is_complete(self):
        return self.complete

    def reset(self):
        self.env.reset()
