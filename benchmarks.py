from stable_baselines3 import PPO, SAC, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from env import CarRacing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

PPO_TRAIN_TIME_STEPS = 2000
DDPG_TRAIN_TIME_STEPS = 10000
SAC_TRAIN_TIME_STEPS = 2000

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
train = 1
if train:

    # set up loggers
    ppo_logger = configure(ppo_log_dir, ["stdout", "csv"])
    sac_logger = configure(sac_log_dir, ["stdout", "csv"])
    ddpg_logger = configure(sac_log_dir, ["stdout", "csv"])

    # Instantiate the env
    ppo_env = CarRacing(render_mode = 'human')
    sac_env = CarRacing(render_mode = 'human')
    ddpg_env = CarRacing(render_mode = 'human')
    

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
env = CarRacing(render_mode = 'human')

PPO_model = PPO.load("./benchmarks/ppo_policy")
SAC_model = SAC.load("./benchmarks/sac_policy")
DDPG_model = DDPG.load("./benchmarks/ddpg_policy")

print("Evaluating PPO")
#PPO_avg_reward = evaluate_policy(PPO_model, env)

print("Evaluating SAC")
SAC_avg_reward = evaluate_policy(SAC_model, env)

print("Evaluating DDPG")
#DDPG_avg_reward = evaluate_policy(DDPG_model, env)

env.close()

#print("\nPPO Average Reward:", PPO_avg_reward)
#print("\nSAC Average Reward:", SAC_avg_reward)
#print("\nDDPG Average Reward:", DDPG_avg_reward)

# Load monitoring data for each algorithm
ppo_monitor_df = pd.read_csv(os.path.join(ppo_log_dir, 'monitor.csv'),skiprows=[0],  index_col=None)
sac_monitor_df = pd.read_csv(os.path.join(sac_log_dir, 'monitor.csv'), skiprows=[0], index_col=None)
ddpg_monitor_df = pd.read_csv(os.path.join(ddpg_log_dir, 'monitor.csv'), skiprows=[0], index_col=None)
td3_df = pd.read_csv('./benchmarks/logs/TD3_log.csv', index_col=None)

# Plot learning curves

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