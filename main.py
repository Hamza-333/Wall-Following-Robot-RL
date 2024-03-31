####### Main loop ##########
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



def evaluate_policy(policy, num_episodes=10):
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

def run_train(policy, env, replay_buffer, max_time, batch_size, start_time, \
              action_noise, seed):
    time = 0
    terminated, truncated = True, True

    # episode vars

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_rewards = []
    episode_rewards_per_step = []
    t0 = t.time()
    time_since_eval = 0
    evals = []
    eval_freq = 5000
    file = "episode-{}"
    dir = './policies'

    while time < max_time:
        # considering both terminated and truncated as end of episode
        if terminated or truncated:
            print("Episode ended")
            if time > 0:
                episode_rewards.append(episode_reward)
                episode_rewards_per_step.append(episode_reward/episode_timesteps)
                # print out stats for the episode
                print("Total Timesteps: {} Episode Num: {} Episode Timesteps: {} Reward: {} Reward Per Step: {}".format(time, episode_num, episode_timesteps, episode_reward, episode_reward/episode_timesteps))
                
                # save policy with current stats 
                # torch.save(policy.state_dict(), '/content/gdrive/My Drive/episode-{}.pk'.format(episode_num))
                if time > start_time:
                    policy.save(file.format(episode_num), dir)
                
                # train policy
                policy.train(episode_timesteps, replay_buffer, batch_size)
            if time_since_eval >= eval_freq:
                time_since_eval = 0
                avg_reward = evaluate_policy(policy)
                evals.append(avg_reward)

                print("Episode Num: {} Average Reward: {}".format(episode_num, avg_reward))
                
                # policy.save("Eval_%d" % (time_since_eval % eval_freq), directory="./")
                # torch.save(policy.state_dict(), '/content/gdrive/My Drive/episode-{}.pk'.format(episode_num))
                if time > start_time:
                    policy.save(file.format(episode_num), dir)
            # reset env after each episode
            state = env.reset(seed = seed)
            terminated, truncated = False, False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # take random actions for first number of timesteps to collect data form env
        if time < start_time:
           action = env.action_space.sample()

        else:
            # now use policy to select action
            action = policy.select_action(np.array(state))
            # Add noise to action to allow agent to explore more states
            # clip between action range
            action = (action + np.random.normal(0, action_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # execute action and retreive observations
            
        next_state, reward, terminated, truncated = env.step(action)    

        term_bool = 0 if episode_timesteps + 1 == 2000 else float(terminated)
        trunc_bool = 0 if episode_timesteps + 1 == 2000 else float(truncated)
        # if time % 100 == 0:
        #     print("State: ", state)
        #     print("Next State: ", next_state)
        #     print("Action: ", action)
        #     print("Reward: ", reward)
        episode_reward += reward

        # add experience to the replay_buffer
        replay_buffer.add(state, next_state, action, reward, term_bool, trunc_bool)
        
        state = next_state
        episode_timesteps += 1
        time_since_eval += 1
        time += 1

    avg_reward = evaluate_policy(policy)
    evals.append(avg_reward)
    policy.save("final_policy", dir)
    # torch.save(policy.state_dict(), '/content/gdrive/My Drive/final-policy.pk')
    # plot episode rewards
    plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
    plt.xlabel = "Episode number"
    plt.ylabel = "Episode reward"
    plt.show()
    # plot evaluations 
    plt.plot([i for i in range(len(evals))], evals)
    plt.xlabel = "Evaluation number"
    plt.ylabel = "Average reward"
    plt.show()

    plt.plot([i for i in range(len(episode_rewards_per_step))], episode_rewards_per_step)
    plt.xlabel = "Episode number"
    plt.ylabel = "Average reward per step"
    plt.show()

if __name__ == "__main__":
    # env = gym.make("CarRacing-v2", continuous=True, render_mode=None)
    env = CarRacing(continuous=True, render_mode='human')
    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    replay_buffer = ReplayBuffer()
    action_dim = env.action_space.shape[0]
    # print(env.action_space.shape[0])
    state_dim = env.observation_space.shape[0]

    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)
    # frequency to evaluate policy
    eval_freq = 2000
    batch_size = 100
    # time to start using policy, before this take random actions
    start_time = 10000
    max_time = 100000

    action_noise = 0.2
    run_train(policy, env, replay_buffer, max_time, batch_size, start_time, action_noise, seed)