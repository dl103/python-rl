#  from pysc2.env import environment
from pysc2.env import sc2_env

STEP_MUL = 16

class Environment:
    def __init__(self):
        env = sc2_env.SC2Env(
                map_name="DefeatZerglingsAndBanelings",
                step_mul=STEP_MUL,
                visualize=True
            )
        print("Init")
