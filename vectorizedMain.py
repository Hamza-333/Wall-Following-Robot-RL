
import numpy as np
import torch
from TD3 import TD3
import argparse
import os
import time
import utils


import gymnasium as gym

import env

MAX_TIME_STEPS = 1000000
max_episode_steps = 2000

NUM_PARALLEL_ENVS = 3
FIN_EPISODES_BEFORE_TRAIN = 4

EXPL_NOISE_REWARD_THRESHOLD = 5000
AVG_REWARD_THRESHOLD = 15500
TAU_REWARD_TRHESHOLD = 5000
MIN_EPS_TIMESTEPS = 300



# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=5):
	avg_reward = 0
	num_fin_episodes = 0
	obs, info = envs.reset()
	avg = 0
	while num_fin_episodes < eval_episodes:
		action = policy.select_vectorized_action(np.array(obs))
		obs, reward, done, _, info = envs.step(action)
		avg_reward += reward
			
        # when an episode ends in any environment
		if info.keys():
			
			finished = info['_final_observation']
			num_fin = np.count_nonzero(finished)
			
			num_fin_episodes += num_fin
			
			avg += np.sum(avg_reward[finished])
			
	avg /= num_fin_episodes
	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (num_fin_episodes, avg))
	print("---------------------------------------")
	return avg

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()

	seed = 0                          # Sets Gym, PyTorch and Numpy seeds
	start_timesteps = 1e4           # How many time steps purely random policy is run for
	eval_freq = 1e4			              # How often (time steps) we evaluate
	max_timesteps = MAX_TIME_STEPS 		# Max time steps to run environment for
	save_models = True			    # Whether or not models are saved
	half_expl_noise = False
	expl_noise=0.01		                # Std of Gaussian exploration noise
	batch_size=256		                # Batch size for both actor and critic
	discount=0.99		                  # Discount factor
	tau=0.001		                    # Target network update rate
	policy_noise=0.1		              # Noise added to target policy during critic update
	noise_clip=0.25	                  # Range to clip target policy noise
	policy_freq=2			                # Frequency of delayed policy updates


	file_name = "TD3_%s" % ( str(seed))
	print("---------------------------------------")
	print ("Settings: %s" % (file_name))
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if save_models and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")
		

	env_id  = 'center_maintaining'
    # Register environment
	env.registerEnv(env_id)
	
	num_envs= NUM_PARALLEL_ENVS
	envs = gym.make_vec(env_id, num_envs=num_envs, render_mode='human')
	
	num_fin_episodes = 0

	# Set seeds
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	state_dim = envs.single_observation_space.shape[0]
	action_dim = envs.single_action_space.shape[0] 
	max_action = float(envs.single_action_space.high[0])

	# Initialize policy
	policy = TD3(state_dim, action_dim, max_action, policy_noise=policy_noise, noise_clip=noise_clip)

	# Load already trained policy
	load_policy = False
	load_init_steps = max_timesteps #10000
	'''
	filename = "Policy_19(1)"
	directory = "./policies"
	policy.load(filename, directory)
	start_timesteps = 0
	'''



	replay_buffer = utils.ReplayBuffer()
	
	# Evaluate untrained policy
	evaluations = []#evaluations = [evaluate_policy(policy)] 

	total_timesteps = 0
	timesteps_since_eval = 0
	train_iteration = 0
	
	all_done = np.full(num_envs, True, dtype=bool)
	
	episode_count = 0
	avg_reward = 0


	t0 = time.time()

	while total_timesteps < max_timesteps:
		#print("total_timesteps:", total_timesteps, end='\r')
		
		if all_done.all(): 
			# calculate average reward over episodes

			if num_fin_episodes!=0: avg_reward /= num_fin_episodes

			num_fin_episodes = 0

	
			if total_timesteps != 0 and (not load_policy or total_timesteps>=load_init_steps):
				
				print("\nTotal T: %d   Train itr: %d   Episodes T: %d   Best Reward: %f   Avg Reward: %f   --  Wallclk T: %d sec" % \
					(total_timesteps, train_iteration, episode_timesteps, max_reward, avg_reward, int(time.time() - t0)))
				
				if avg_reward >= AVG_REWARD_THRESHOLD:
					print("\n\nAvg Reward Threshold Met -- Training Terminated\n")
					break

				if avg_reward >= TAU_REWARD_TRHESHOLD:
					tau = 0.001
					print("\n\n\nHalving Tau to %f \n\n\n" % tau)


                # half exploration noise 
				if half_expl_noise and avg_reward >= EXPL_NOISE_REWARD_THRESHOLD:
					expl_noise = expl_noise / 2
					print("\n\n\nHalving expl noise to %f \n\n\n" % expl_noise)

				
				policy.save("Policy_%d" % (train_iteration), directory="./policies")
				
				if episode_timesteps < MIN_EPS_TIMESTEPS:
					policy.train(MIN_EPS_TIMESTEPS, replay_buffer, tau, batch_size)
				else:
					print("STANDARDIZED TRAINING ITERATIONS")
					policy.train(episode_timesteps, replay_buffer, tau, batch_size)
			
			# Evaluate episode
			if timesteps_since_eval >= eval_freq:
				timesteps_since_eval %= eval_freq
				eval_score = evaluate_policy(policy)
				evaluations.append(eval_score)
				
				if save_models: policy.save(file_name, directory="./pytorch_models")
				np.save("./results/%s" % (file_name), evaluations) 
			
			# Reset environment
			obs, info = envs.reset(seed=[seed + i for i in range(num_envs)])
			seed+=num_envs
			all_done = np.full(num_envs, False, dtype=bool)
			episode_reward = np.zeros(num_envs, dtype=float)
			episode_timesteps = 0
			train_iteration += 1 
			
			max_reward = None
			avg_reward = 0
		
		# Select action randomly or according to policy
		if total_timesteps == start_timesteps:
			print("\n\n\nPolicy actions started\n\n\n")

		if total_timesteps < start_timesteps:
			# Random actions for each environment
			action = envs.action_space.sample()
		else:
			action = policy.select_vectorized_action(obs)
			
			if expl_noise != 0: 
				#TODO
				action = (action + np.random.normal(0, expl_noise, size=envs.single_action_space.shape[0])).clip(envs.single_action_space.low, envs.single_action_space.high)
			

		# Perform action
		new_obs, reward, done, truncated, info = envs.step(action) 
		episode_reward += reward
        # when an episode ends in any environment
		if info.keys():
			
			finished = info['_final_observation']
			num_fin = np.count_nonzero(finished)
			
			num_fin_episodes += num_fin
			episode_count += num_fin
			
            # all_done marks the environments whose episodes ended
			all_done = np.logical_or(all_done, finished)
			
			print("Episode%d reward for finished enviroments:" % episode_count, episode_reward[finished])

            #Set min reward among finished episodes
			if max_reward is not None:
				max_reward = max(max_reward, max(episode_reward[finished]))
			else:
				max_reward = max(episode_reward[finished])

			avg_reward += sum(episode_reward[finished])
			
			#set episode reward for respective environments 0
			episode_reward[finished] = 0

			


		done_bool = np.full(num_envs, False, dtype=bool) if episode_timesteps + 1 == max_episode_steps else all_done
		
		

		# Store data in replay buffer
		for i in range(num_envs):
			if info.keys() and info['_final_observation'][i] == True:
				replay_buffer.add(obs[i], info['final_observation'][i], action[i], reward[i], float(all_done[i]))
			else:
				replay_buffer.add(obs[i], new_obs[i], action[i], reward[i], float(all_done[i]))

		obs = new_obs

        #   Episode time_steps for all episodes in each environment
		episode_timesteps += num_envs
		total_timesteps += num_envs
		timesteps_since_eval += num_envs


	# Final evaluation 
	evaluations.append(evaluate_policy(policy))
	if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	np.save("./results/%s" % (file_name), evaluations) 




