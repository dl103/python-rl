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
        prev_state = current_state
        prev_action = current_action
        current_action = agent.get_action()
        env.step(current_action)
        current_state = env.observation
        reward = env.reward

        # Update network/collect rewards
        if env.is_complete():
            target = reward
        else:
            target = reward + gamma * nn.max_output(current_state)

        # target is the reward + q value of best action

        prev_state_vector = nn.predict(prev_state)
        prev_state[0][prev_action] = target
        nn.update(prev_state, prev_state_vector)

        # Reset if complete
        if env.is_complete():
            env.reset()
            current_state = env.observation
            current_action = None
            scores.append(score)
            if len(scores) % 50 == 0:
                print("Score Average: " + str(sum(scores)/len(scores)))
                scores = []
            score = 0
        score += 1

if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
