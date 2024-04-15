from stable_baselines3 import PPO, SAC, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from env import CarRacing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

PPO_TRAIN_TIME_STEPS = 10000
DDPG_TRAIN_TIME_STEPS = 10000
SAC_TRAIN_TIME_STEPS = 10000

LEARNING_RATE = 0.001

def evaluate_policy(policy, env, num_episodes=1):
    total_reward = 0
    for i in range(num_episodes):
        state, info = env.reset()
        next_state = state
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = policy.predict(np.array(next_state), deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action[0])
            total_reward += reward

    avg_reward = total_reward / num_episodes

    return avg_reward

# logging directories for each algorithm

#Monitor
ppo_log_dir = "./benchmarks/logs/ppo_logs/"
sac_log_dir = "./benchmarks/logs/sac_logs/"
ddpg_log_dir = "./benchmarks/logs/ddpg_logs/"


######## Training ########

parser = argparse.ArgumentParser(description='Settings for env')
parser.add_argument("--train", default=False)
args = parser.parse_args()


train = args.train
if train:
    # set up loggers
    ppo_logger = configure(ppo_log_dir, ["stdout", "csv"])
    sac_logger = configure(sac_log_dir, ["stdout", "csv"])
    ddpg_logger = configure(sac_log_dir, ["stdout", "csv"])

    # Instantiate the env
    ppo_env = CarRacing(render_mode = None)
    sac_env = CarRacing(render_mode = None)
    ddpg_env = CarRacing(render_mode = None)
    

    # Create monitor wrappers for each algorithm with unique logging directories
    ppo_env = Monitor(ppo_env, ppo_log_dir)
    sac_env = Monitor(sac_env, sac_log_dir)
    ddpg_env = Monitor(ddpg_env, ddpg_log_dir)

    # PPO model
    PPO_model = PPO("MlpPolicy", ppo_env, verbose = 1, learning_rate= LEARNING_RATE)
    PPO_model.set_logger(ppo_logger)

    print("Training PPO")
    PPO_model.learn(total_timesteps=PPO_TRAIN_TIME_STEPS)
    PPO_model.save("./benchmarks/ppo_policy")

    # SAC
    SAC_model = SAC("MlpPolicy", sac_env, verbose=1, learning_rate=LEARNING_RATE)
    SAC_model.set_logger(sac_logger)
    
    print("Training SAC")
    SAC_model.learn(total_timesteps=SAC_TRAIN_TIME_STEPS)
    SAC_model.save("./benchmarks/sac_policy")

    # DDPG
    # The noise objects for DDPG
    n_actions = ddpg_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

    DDPG_model = DDPG("MlpPolicy", ddpg_env, action_noise=action_noise, verbose=1, tau=0.001)
    DDPG_model.set_logger(ddpg_logger)
    
    print("Training DDPG")
    DDPG_model.learn(total_timesteps=DDPG_TRAIN_TIME_STEPS)
    DDPG_model.save("./benchmarks/ddpg_policy")

    print("Training complete")

    ppo_env.close()
    sac_env.close()
    ddpg_env.close()



######## Evaluation ########

# Load monitoring data for each algorithm
ppo_monitor_df = pd.read_csv(os.path.join(ppo_log_dir, 'monitor.csv'),skiprows=[0],  index_col=None)
sac_monitor_df = pd.read_csv(os.path.join(sac_log_dir, 'monitor.csv'), skiprows=[0], index_col=None)
ddpg_monitor_df = pd.read_csv(os.path.join(ddpg_log_dir, 'monitor.csv'), skiprows=[0], index_col=None)

td3_df = pd.read_csv('./benchmarks/logs/TD3_log.csv', index_col=None)

# create column for timesteps 
ppo_monitor_df['timesteps'] = ppo_monitor_df['l'].cumsum()
sac_monitor_df['timesteps'] = sac_monitor_df['l'].cumsum()
ddpg_monitor_df['timesteps'] = ddpg_monitor_df['l'].cumsum()
td3_df['timesteps'] = td3_df['l'].cumsum()


# Plot learning curves

# Rewards vs timesteps
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['timesteps'], ppo_monitor_df['r'], label='PPO')
plt.plot(sac_monitor_df['timesteps'], sac_monitor_df['r'], label='SAC')
plt.plot(ddpg_monitor_df['timesteps'], ddpg_monitor_df['r'], label='DDPG')
plt.plot(td3_df['timesteps'], td3_df['r'], label='TD3')
plt.xlabel('Timesteps')
plt.ylabel('r')
plt.title('Reward vs Timesteps')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()


# Rewards vs Episodes
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['r'], label='PPO')  
plt.plot(sac_monitor_df['r'], label='SAC')  
plt.plot(ddpg_monitor_df['r'], label='DDPG') 
plt.plot(td3_df['r'], label='TD3')
plt.xlabel('Episodes')
plt.ylabel('Episode Reward')
plt.title('Learning Curves: Rewards vs Episodes')
plt.xticks(rotation=90) 
plt.legend()
plt.show()

# Episode length vs Episodes
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['l'], label='PPO') 
plt.plot(sac_monitor_df['l'], label='SAC')  
plt.plot(ddpg_monitor_df['l'], label='DDPG')  
plt.plot(td3_df['l'], label='TD3')
plt.xlabel('Episodes')
plt.ylabel('Episode length')
plt.title('Learning Curves: Episode len vs Episodes')
plt.xticks(rotation=90)  
plt.legend()
plt.show()

# Rewards + Episode len vs Episodes
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['l'] + ppo_monitor_df['r'], label='PPO')  
plt.plot(sac_monitor_df['l'] + sac_monitor_df['r'], label='SAC')  
plt.plot(ddpg_monitor_df['l'] + ddpg_monitor_df['r'], label='DDPG') 
plt.plot(td3_df['l'] + td3_df['r'], label='TD3')
plt.xlabel('Episodes')
plt.ylabel('Episode len + Rewards')
plt.title('Learning Curves: Episode len + Rewards vs Episodes')
plt.xticks(rotation=90)  # Rotate the y-axis labels by 45 degrees
plt.legend()
plt.show()


# Rewards vs Episodes (upto 48 episodes)
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['r'][:48], label='PPO')  
plt.plot(sac_monitor_df['r'][:48], label='SAC')  
plt.plot(ddpg_monitor_df['r'][:48], label='DDPG') 
plt.plot(td3_df['r'], label='TD3')
plt.xlabel('Episodes')
plt.ylabel('Episode Reward')
plt.title('Learning Curves: Rewards vs Episodes')
plt.xticks(rotation=90) 
plt.legend()
plt.show()

# Episode length vs Episodes (upto 48 episodes)
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['l'][:48], label='PPO') 
plt.plot(sac_monitor_df['l'][:48], label='SAC')  
plt.plot(ddpg_monitor_df['l'][:48], label='DDPG')  
plt.plot(td3_df['l'], label='TD3')
plt.xlabel('Episodes')
plt.ylabel('Episode length')
plt.title('Learning Curves: Episode len vs Episodes')
plt.xticks(rotation=90)  
plt.legend()
plt.show()

# Rewards + Episode len vs Episodes (upto 48 episodes)
plt.figure(figsize=(10, 5))
plt.plot(ppo_monitor_df['l'][:48] + ppo_monitor_df['r'][:48], label='PPO')  
plt.plot(sac_monitor_df['l'][:48] + sac_monitor_df['r'][:48], label='SAC')  
plt.plot(ddpg_monitor_df['l'][:48] + ddpg_monitor_df['r'][:48], label='DDPG') 
plt.plot(td3_df['l'][:48] + td3_df['r'][:48], label='TD3')
plt.xlabel('Episodes')
plt.ylabel('Episode len + Rewards')
plt.title('Learning Curves: Episode len + Rewards vs Episodes')
plt.xticks(rotation=90)  # Rotate the y-axis labels by 45 degrees
plt.legend()
plt.show()