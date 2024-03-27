from Actor import Actor
from Critic import Critic
from utils import ReplayBuffer
import gymnasium as gym


def run_train(policy, env, replay_buffer):
    steps = 0



if __name__ == "__main__":
    env = gym.make("CarRacing-v2", continuous=True)
    replay_buffer = ReplayBuffer()
    policy = TD3

