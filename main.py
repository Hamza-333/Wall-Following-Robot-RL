import numpy as np
import torch
from TD3 import TD3
import time

import utils
from utils import SEED

import gymnasium as gym
import env

import argparse

from env import CarRacing
import argparse
import os, csv

MAX_TIME_STEPS = 1000000
max_episode_steps = 2000

#Options to change expl noise and tau
LOWER_EXPL_NOISE = {"On" : True, "Reward_Threshold":14000, 'Value': 0.001}
LOWER_TAU = {"On" : True, "Reward_Threshold":18000, 'Timesteps_Threshold' : 20000, 'Value': 0.00075}

#load already trained policy
LOAD_POLICY = {"On": False, 'init_time_steps': 1e4}

#Avg reward termination condition
AVG_REWARD_TERMIN_THRESHOLD = 19500
# Time steps below which a standard training iteration param is passed
MIN_EPS_TIMESTEPS = 500

# Specify the file name
LOGS_FILEPATH = './benchmarks/logs/TD3_log.csv'
if not os.path.exists('./benchmarks/logs/'):
    os.makedirs('./benchmarks/logs/')

with open(LOGS_FILEPATH, 'w', newline='') as file:
    log_writer = csv.writer(file)

    # Write headings
    log_writer.writerow(['r', 'l'])


def evaluate_policy(policy, num_episodes=10):
    total_reward = 0
    for i in range(num_episodes):
        state, info = env.reset()
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

    parser = argparse.ArgumentParser(description='Settings for env')

    parser.add_argument("--var_speed", default=False)					
    parser.add_argument("--accel_brake", default=False)
    parser.add_argument("--render_mode", default=None)
    args = parser.parse_args()

                
    start_timesteps = 1e3           	# How many time steps purely random policy is run for
    eval_freq = 1e4			             # How often (time steps) we evaluate
    max_timesteps = MAX_TIME_STEPS 		# Max time steps to run environment for
    save_models = True			    	# Whether or not models are saved

    expl_noise=0.2	                # Std of Gaussian exploration noise
    batch_size=256		                # Batch size for both actor and critic
    tau=0.001		                    # Target network update rate
    policy_noise=0.1		              # Noise added to target policy during critic update
    noise_clip=0.25	                  # Range to clip target policy noise


    file_name = "TD3_%s" % ( str(SEED))
    print("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = CarRacing(render_mode=args.render_mode, var_speed=args.var_speed, accel_brake = args.accel_brake)

    #Counter to track finished episode within one iteration of parallel runs
    num_fin_episodes = 0

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action, policy_noise=policy_noise, noise_clip=noise_clip)

    # Load already trained policy
    if LOAD_POLICY["On"]:
        filename = "Policy_19(1)"
        directory = "./policies"
        policy.load(filename, directory)
        start_timesteps = 0

    # Init replay buffer
    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    evaluations = []#evaluations = [evaluate_policy(policy)] 

    total_timesteps = 0 
    timesteps_since_eval = 0
    train_iteration = 0

    t0 = time.time()
    done = True

    episode_reward = 0

    while total_timesteps < max_timesteps:
        
        if done: 

            if total_timesteps != 0 and (not LOAD_POLICY['On'] or total_timesteps>=LOAD_POLICY["init_time_steps"]):
                
                print("\nData Stats:\nTotal T: %d   Train itr: %d   Episodes T: %d  Reward: %f   --  Wallclk T: %d sec" % \
                    (total_timesteps, train_iteration, episode_timesteps, episode_reward, int(time.time() - t0)))
                
                # Store metrics
                with open(LOGS_FILEPATH, 'a', newline='') as file:
                    log_writer = csv.writer(file)
                    log_writer.writerow([reward, episode_timesteps])

                if reward >= AVG_REWARD_TERMIN_THRESHOLD:
                    print("\n\nAvg Reward Threshold Met -- Training Terminated\n")
                    break
                    
                # Lower learning rate 
                if LOWER_TAU["On"] and reward >= LOWER_TAU["Reward_Threshold"] and total_timesteps>=LOWER_TAU["Timesteps_Threshold"]:
                    tau = LOWER_TAU["Value"]
                    print("\n-------Lowered Tau to %f \n" % LOWER_TAU["Value"])
                    LOWER_TAU["On"] = False

                # Lower exploration noise 
                if LOWER_EXPL_NOISE["On"] and reward >= LOWER_EXPL_NOISE["Reward_Threshold"]:
                    expl_noise = LOWER_EXPL_NOISE["Value"]
                    print("\n-------Lowered expl noise to %f \n" % LOWER_EXPL_NOISE["Value"])
                    LOWER_EXPL_NOISE["On"] = False

                # save each policy with above stats before training
                policy.save("Policy_%d" % (train_iteration), directory="./policies")

                print("\nTraining: ", end=" ")
                if episode_timesteps < MIN_EPS_TIMESTEPS:
                    print("STANDARDIZED TRAINING ITERATIONS")
                    policy.train(MIN_EPS_TIMESTEPS, replay_buffer, tau, batch_size)
                else:
                    policy.train(episode_timesteps, replay_buffer, tau, batch_size)
                
                print("-Finished ")
                print("\n-----------------------")
            
            # Evaluate episode
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                eval_score = evaluate_policy(policy)
                evaluations.append(eval_score)

                if save_models: policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations) 
            
            # Reset environment
            print("\nCollecting data:")
            
            obs, info = env.reset(seed=SEED)
            episode_timesteps = 0
            train_iteration += 1 
            episode_reward = 0
            
            max_reward = None
            
        # Select action randomly or according to policy
        if total_timesteps == start_timesteps:
            print("\n\n\nPolicy actions started\n\n\n")

        if total_timesteps < start_timesteps:
            # Random actions for each environment
            action = env.action_space.sample()
        else:
            action = policy.select_action(obs)
            
            if expl_noise != 0: 
                action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        replay_buffer.add(obs, new_obs, action, reward, float(done))
        
        obs = new_obs
        
        #   Episode time_steps for all episodes in each environment
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation 
    evaluations.append(evaluate_policy(policy))
    if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations) 

    env.close()
