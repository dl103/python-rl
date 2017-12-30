import pdb
import numpy as np
import sys

from environments.openai_environment import OpenAIEnvironment
from algorithms.deep_q_learning import DeepQLearning

from absl import flags

def main():
    """Cartpole example"""
    # Load environment
    env = OpenAIEnvironment('CartPole-v0')

    algorithm = DeepQLearning(gamma = 0.95, epsilon = 0.01, num_actions = env.num_actions(), observation_dim=env.observation_dimensions)
    # Train
    train(env, algorithm)

# Main function to train the
# env - OpenAI Gym Environment
def train(env, algorithm):
    score = 0
    scores = []

    while True:
        algorithm.act(env)

        # Reset if complete
        if env.is_complete():
            env.reset()
            scores.append(score)
            if len(scores) % 50 == 0:
                print("Score Average: " + str(sum(scores)/len(scores)))
                scores = []
            score = 0
        score += 1

if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
