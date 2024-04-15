
import numpy as np
import TD3
from env import CarRacing
import argparse

#process file input
parser = argparse.ArgumentParser(description='Settings for env')
parser.add_argument("--load_policy", default="10")
parser.add_argument("--load_model", default=None)
parser.add_argument("--var_speed", default=False)		
parser.add_argument("--accel_brake", default=False)	


args = parser.parse_args()

#Init env 
env = CarRacing(
    render_mode="human",
    var_speed=args.var_speed,					# train for variable speeds
	accel_brake = args.accel_brake)				# train for acceleration and brake

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

print(state_dim, action_dim)

# Initialize policy
policy = TD3.TD3(state_dim, action_dim, max_action)

#Load policy or model based on input
if args.load_policy:
    filename = "Policy_" + str(args.load_policy)
    directory = "./policies"
    policy.load(filename, directory)
else:
    filename = "TD3_" + args.load_model
    directory = "./pytorch_models"
    policy.load(filename, directory)

# Reset Env
done = False
state, info = env.reset()

total_reward = 0
total_reward_per_tile = 0
cte_list = []

num_sim = 10
rewards = []
for i in range(num_sim):
    
    #Simulation loop
    done = False
    while not done:
        # Select action
        action = policy.select_action(np.array(state))
        action = [action[0], 0, 0]

        # Perform action
        state, reward, terminated, truncated, info = env.step(action) 

        # account for cte 
        cte_list.append(state[1])

        # account for total rewards
        total_reward += reward
        rewards.append(reward)
        if  terminated or truncated:
            # account rewardPerTile
            total_reward_per_tile += info["rewardPerTile"]

            # reset
            state, info = env.reset()

            done = True
        
print("Variance of CTE: ", np.var(cte_list)) * env.road_half_width

print("Average reward: ", total_reward / num_sim)

print("Average reward per action: ", np.sum(rewards)/len(rewards))

print("Average tile reward: ", total_reward_per_tile / num_sim)

print("Average CTE: ", np.sum(np.abs(cte_list))/len(cte_list))




