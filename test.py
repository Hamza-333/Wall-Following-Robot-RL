from Actor import Actor
from Critic import Critic
from utils import ReplayBuffer
import torch
import numpy as np
import time as t
from TD3 import TD3
import gymnasium as gym
import matplotlib.pyplot as plt
from car_racing import CarRacing



def evaluate_policy(policy, num_episodes=1):
    total_reward = 0
    for i in range(num_episodes):
        state = env.reset()
        next_state = state
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = policy.select_action(np.array(next_state))
            next_state, reward, terminated, truncated = env.step(action)
            total_reward += reward

    avg_reward = total_reward / num_episodes

    print("----------------------------")
    print("Average policy reward over %d episodes: %f" % (num_episodes, avg_reward))
    print("----------------------------")
    return avg_reward

if __name__ == "__main__":

    env = CarRacing(continuous=True, render_mode='human')
    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer = ReplayBuffer()
    action_dim = env.action_space.shape[0]

    state_dim = env.observation_space.shape[0]

    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)
    file = "episode-150"
    dir = "./best-policies"
    policy.load(file, dir)

    evaluate_policy(policy)