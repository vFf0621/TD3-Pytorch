#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:29:38 2022

@author: guanfei1
"""


import torch
from torch import optim
from torch import nn
from collections import deque
import gym
import random
import numpy as np
class Actor(nn.Module):
    def __init__(self, env, hidden = 300, lr = 0.001):
        super().__init__()
        self.linear1 = nn.Linear(env.observation_space.shape[0], 
                                           hidden + 100)

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100, hidden)
        self.linear3 = nn.Linear(hidden, env.action_space.shape[0])
        self.tanh = nn.Tanh()
        self.optim = optim.Adam(self.parameters(), lr = lr)
    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x

    
class Critic(nn.Module):
    def __init__(self, env, hidden=300, lr = 0.001):
        super().__init__()
        self.linear1= nn.Linear(env.observation_space.shape[0]+ env.action_space.shape[0], 
                                           hidden + 100)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(hidden + 100 , hidden)
                                 
        self.linear3 = nn.Linear(hidden, 1)
        self.optim = optim.Adam(self.parameters(), lr = lr)


    def forward(self, state, action):
        if len(action.shape) < len(state.shape):
            action = action.unsqueeze(-1)
        x = self.linear1(torch.cat([state, action], dim=1))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)

        return x
  
class TD3:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if\
        torch.cuda.is_available() else "cpu")
        self.actor = Actor(env=self.env).to(self.device)
        self.critic1 = Critic(env=self.env).to(self.device)
        self.critic2 = Critic(env=self.env).to(self.device)

        self.target_actor = Actor(env=self.env).to(self.device)
        self.target_critic1 = Critic(env=self.env).to(self.device)
        self.target_critic2 = Critic(env=self.env).to(self.device)
        self.update_int=2
        self.gamma = 0.99
        self.tau = 0.005
        self.actor.load_state_dict(self.target_actor.state_dict())
        self.critic1.load_state_dict(self.target_critic1.state_dict())
        self.critic2.load_state_dict(self.target_critic2.state_dict())

        self.replay_buffer = deque(maxlen=1000000)
        self.loss = torch.nn.MSELoss()
        self.reward_buffer = deque(maxlen=100)
        self.action_high = torch.from_numpy(self.env.action_space.high).\
        to(self.device)
        s = env.reset()[0]
        self.count = 0
        for i in range(100):
            done = False
            while not done:
                action = self.env.action_space.sample()
                s_, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.append((s, action, s_, reward, done))
                s = s_
    def act(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        state = state.float().to(self.device)
        
        return torch.clamp(self.actor(state)+torch.normal(mean=torch.tensor([0.]),
            std=torch.tensor([0.1])).to(self.device), -self.action_high, 
            self.action_high).cpu().detach().numpy()
    def soft_update(self):
        for param, target_param in zip(self.critic1.parameters(),
                                     self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + \
                               (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(),
                                     self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + \
                               (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), 
                      self.target_actor.parameters()):
           target_param.data.copy_(self.tau * param.data + 
                              (1 - self.tau) * target_param.data)
    def sample(self, batch_size = 100):
        t = random.sample(self.replay_buffer, batch_size)
        actions = []
        states = []
        dones = []
        states_ = []
        rewards = []
        for i in t:
            state, action, state_, reward, done  = i
            
            states.append(state)

            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            states_.append(state_)
        
        states = torch.from_numpy(np.array(states)).\
        to(self.device).float()
        actions = torch.from_numpy(np.array(actions)).\
        to(self.device).float()
        rewards = torch.from_numpy(np.array(rewards)).\
            to(self.device).float()
        dones = torch.from_numpy(np.array(dones)).\
            to(self.device)
        states_ = torch.from_numpy(np.array(states_)).to(self.device).float()
        
        return states, actions, states_, rewards, dones
        
    def train(self):
    
        states, actions, states_, rewards, dones = self.sample()
        target_action = torch.clamp(self.target_actor(states_) + \
        torch.clamp(torch.normal(mean=torch.tensor([0.]), std=torch.tensor([0.2])),
                   -0.4, 0.4).to(self.device), -self.action_high, self.action_high)
        q1_ = self.target_critic1(states_, target_action).view(-1)
        q2_ = self.target_critic2(states_, target_action).view(-1)
        q1_[dones] = 0.
        q2_[dones] = 0.
        q_ = torch.min(q1_, q2_)
        target_q = rewards + self.gamma * q_

        q1 = self.critic1(states, actions).view(-1)
        q2 = self.critic2(states, actions).view(-1)

        self.critic1.optim.zero_grad()
        self.critic2.optim.zero_grad()

        loss1 = self.loss(target_q, q1)
        loss2 = self.loss(target_q, q2)
        loss = loss1 + loss2
        loss.backward()
        self.critic1.optim.step()

        self.critic2.optim.step()

        
        if self.count % self.update_int == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor.optim.zero_grad()
            actor_loss.backward()
            self.actor.optim.step()
            self.soft_update()
        self.count += 1
