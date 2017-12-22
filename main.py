import sys
from a3c.environment import Environment

def main(argv):
    Environment().run()

if __name__ == "__main__":
    main(sys.argv)
