import sys
import pdb
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

    for i in range(1000):
        # Perform action from network
        env.step(agent.get_action())

        # Update network/collect rewards

        # Reset if complete
        if env.is_complete():
            env.reset()

if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
