from Actor import Actor
from Critic import Critic
from utils import ReplayBuffer

from TD3 import TD3
import gymnasium as gym


def run_train(policy, env, replay_buffer):
    steps = 0



if __name__ == "__main__":
    env = gym.make("CarRacing-v2", continuous=True)
    replay_buffer = ReplayBuffer()
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space
    max_action = float(env.action_space.high[0])

    policy = TD3(action_dim, state_dim, max_action)


