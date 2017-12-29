import sys
import pdb
import numpy as np

from common.environment import Environment
from deep_q.network import Network
from deep_q.agent import Agent
from absl import flags

def main():
    # Load environment
    env = Environment()
    # Input size, output size
    nn = Network(env.observation_space().shape[0], env.action_space().n, 0.001)

    # Train
    train(env, nn)

# Main function to train the
# env - OpenAI Gym Environment
# nn - Network
def train(env, nn):
    print("Training")
    # Init single agent
    agent = Agent(env, nn, 0.01)
    current_state = env.observation
    current_action = None
    gamma = 0.95
    score = 0
    scores = []

    while True:
        # Perform action from network
        current_state = env.observation
        current_action = agent.get_action()
        env.step(current_action)
        next_state = env.observation
        reward = env.reward

        # Update network/collect rewards
        if env.is_complete():
            target = reward
        else:
            target = reward + gamma * nn.max_output(next_state)

        # Take the entire action vector of the "current_state" and update the
        # action that we took to the target reward. This way, all actions that
        # we didn't take will have an error of 0 when we subtract out the Q
        # estimation.
        current_state_vector = nn.predict(current_state)
        current_state_vector[0][current_action] = target
        nn.update(current_state, current_state_vector)

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
