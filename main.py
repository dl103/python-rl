import sys
import pdb
from common.environment import Environment
from a3c.network import Network
from absl import flags

def main():
    # Load environment
    env = Environment()
    # Input size, output size
    nn = Network(env.observation_space().shape[0], env.action_space().n, 0.001)

    # Train
    train(env)

# Main function to train the
def train(env):
    print("Training")
    # Init single agent

    for i in range(1000):
        # Perform action from network
        env.step(env.action_space().sample())
        if env.is_complete():
            env.reset()

        # Update network/collect rewards

if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
