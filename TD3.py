from Actor import Actor 
from Critic import Critic
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
    def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
  
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def select_vectorized_action(self, states):
        states_tensor = torch.FloatTensor(states).to(device)
        actions = self.actor(states_tensor)
        return actions.cpu().data.numpy()
    
    def train(self, iterations, replay_buffer, batch_size=256):
        for i in range(iterations):
            self.total_it += 1
            
            s, ns, ac, r, d = replay_buffer.sample(batch_size)
            
            state = torch.FloatTensor(s).to(device)
            next_state = torch.FloatTensor(ns).to(device)
            action = torch.FloatTensor(ac).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            
            with torch.no_grad():
			    # For next action,  consider the policy and add noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = (
				    self.actor_target(next_state) + noise
			    ).clamp(-self.max_action, self.max_action)

			    # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                min_target = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * self.discount * min_target)

            # Get current Q estimates
            curr_Q1, curr_Q2 = self.critic(state, action)
            
            # Calculate loss for crtic 
            mse_loss_1 = nn.MSELoss()
            mse_loss_2 = nn.MSELoss()
            loss_for_critic =  mse_loss_1(curr_Q1, target_Q) + mse_loss_2(curr_Q2, target_Q)
            
            # Optimization for the critic 
            self.critic_optimizer.zero_grad()
            loss_for_critic.backward()
            self.critic_optimizer.step()
            
            # update the policy
            if i % self.policy_freq == 0:
                loss_for_actor = -self.critic.q2(state, self.actor(state)).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                loss_for_actor.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))     
      
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=torch.device('cpu')))