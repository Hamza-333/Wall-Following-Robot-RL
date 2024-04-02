from stable_baselines3 import PPO, SAC, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from env import CarRacing
import numpy as np
from utils import evaluate_policy
import matplotlib.pyplot as plt

# Instantiate the env
env = CarRacing(render_mode = 'human')

# PPO model
PPO_model = PPO("MlpPolicy", env, verbose = 1)
PPO_model.learn(total_timesteps=10000)
PPO_model.save("./benchmarks/ppo_policy")

# SAC
SAC_model = SAC("MlpPolicy", env, log_interval=4)
SAC_model.learn(total_timesteps=10000)
SAC_model.save("./benchmarks/sac_policy")

# DDPG
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

DDPG_model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
DDPG_model.learn(total_timesteps=10000, log_interval=10)
DDPG_model.save("./benchmarks/ddpg_policy")



# Evaluate models
PPO_avg_reward = evaluate_policy(PPO_model, env)
SAC_avg_reward = evaluate_policy(SAC_model, env)
DDPG_avg_reward = evaluate_policy(DDPG_model, env)

print("PPO Average Reward:", PPO_avg_reward)
print("SAC Average Reward:", SAC_avg_reward)
print("DDPG Average Reward:", DDPG_avg_reward)

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(PPO_model.episode_rewards, label='PPO')
plt.plot(SAC_model.episode_rewards, label='SAC')
plt.plot(DDPG_model.episode_rewards, label='DDPG')
plt.xlabel('Episodes')
plt.ylabel('Episode Reward')
plt.title('Learning Curves')
plt.legend()
plt.show()