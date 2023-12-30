


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:04:16 2022

@author: guanfei1
"""
import collections
import gymnasium as gym
import numpy as np
from TD3 import *
import matplotlib.pyplot as plt
import wandb

# Initialize variables
x_est = 0.0  # initial state estimate
P_est = 1.0  # initial estimate covariance
Q = 0.1      # process noise covariance
R = 1      # measurement noise covariance


if __name__ == '__main__':
    env = gym.make('Ant-v2')
    project_name = 'TD3'
    wandb.init(project=project_name, entity='fguan06', settings=wandb.Settings(start_method="thread"))
    wandb.run.name = "TD3-Ant-v2"

    agent = TD3(env)
    ys = []
    eps = 2000
    ret = torch.tensor(0.).to(agent.device).view(-1)
    xs = list(range(eps))
    step = 0
    print("Replay Buffer Initialized")
    eva = False
    b = 0
    eps = 0
    loss = 0
    prev_r = -99999
    max_dev = -99999
    term = 0
    x_est = 0 # initial state estimate

    while not step > 1000000:
        done = False
        trunc = False
        episode_reward = 0
        s = env.reset()[0]
        ret_lis = []
        eps += 1
        state_list = []
        action_list = []
        P_est = 1.0  # initial estimate covariance
        lambd_lis = []
        xs = []
        while not (trunc or done):
            state_list.append(s)

            action = agent.act(s, eva)
            target_action = torch.clamp(agent.target_actor(torch.from_numpy(s).to(agent.device).float()) + \
            torch.clamp(torch.normal(mean=torch.tensor([0.]), std=torch.tensor([0.2])),
                   -0.5, 0.5).to(agent.device), -agent.action_high, agent.action_high).detach().cpu().tolist()
            action_list.append(target_action)
            s_, r, done, trunc, _ = env.step(action)

            agent.replay_buffer.append((s, action, s_, r, done))
            s = s_
            episode_reward += r
            step += 1
            if len(agent.replay_buffer) > agent.batch_size :
                eva = False
                agent.train()

            ret_lis.append(r)
            if done:
                break 
        ret_lis[-1] = torch.FloatTensor(ret_lis[-100:]).mean()/(1-agent.gamma)
        
        for k in reversed(range(len(ret_lis)-1)):
            ret_lis[k] = ret_lis[k] + agent.gamma* ret_lis[k+1]
        plt.plot(list(range(len(ret_lis))), ret_lis)
        x_est =ret_lis[0]
        plt.plot(list(range(len(ret_lis))), ret_lis)

        ret_lis = torch.Tensor(ret_lis).to(agent.device).float()
        #agent.lambda_opt.zero_grad()
        #loss = nn.SmoothL1Loss()(Q_, ret_lis.detach())
        #loss.backward()
        d = {}

        
        #d["lambd_grad_norm"] = torch.nn.utils.clip_grad_norm_([agent.lambd], 100).mean()
        #agent.lambda_opt.step()
        ma = (torch.max(*agent._get_target_values(torch.Tensor(state_list).to(agent.device).float(), torch.Tensor(action_list).to(agent.device).float())))
                
        d["lambd_loss"] = loss
        d["Q_max"] = ma.mean()
        d["Q_max_var"] = ma.var()
        mi = (torch.min(*agent._get_target_values(torch.Tensor(state_list).to(agent.device).float(), torch.Tensor(action_list).to(agent.device).float())))
        d["Q_min"] = mi.mean()
        d["Q_min_var"] = mi.var()
        d["Q_diff"] = d["Q_max"] - d["Q_min"]
        d["return"] = episode_reward
        d["lambd"] = agent.lambd.item()
        d["graph"] = plt
        wandb.log(step=step, data=d)
        agent.reward_buffer.append(episode_reward)
        mean = np.mean(agent.reward_buffer)
        ys.append(mean)
        
        print("Episode", eps, "Episode Reward", episode_reward, ", Average Reward", 
              mean)
    plt.plot(xs, ys)
    
    
    
        
