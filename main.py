

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
R = 300      # measurement noise covariance

def KF(x_meas, x_est, P_est):
    x_pred = x_est  # since in 1D, the state does not change if no control input is given
    P_pred = P_est + Q

    K = P_pred / (P_pred + R)  # Kalman Gain
    x_est = x_pred + K * (x_meas - x_pred)
    P_est = (1 - K) * P_pred
    return x_est, P_est

if __name__ == '__main__':
    env = gym.make('Walker2d-v2')
    project_name = 'TD3'
    wandb.init(project=project_name, entity='fguan06', settings=wandb.Settings(start_method="thread"))
    wandb.run.name = "TD3-Walker2D"

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

    while not step > 1000000:
        done = False
        episode_reward = 0
        s = env.reset()[0]
        ret_lis = []
        eps += 1
        state_list = []
        action_list = []
        x_est = 0.0  # initial state estimate
        P_est = 1.0  # initial estimate covariance
        lambd_lis = []
        xs = []
        for i in range(1000):
            state_list.append(s)

            action = agent.act(s, eva)
            target_action = torch.clamp(agent.target_actor(torch.from_numpy(s).to(agent.device).float()) + \
            torch.clamp(torch.normal(mean=torch.tensor([0.]), std=torch.tensor([0.2])),
                   -0.5, 0.5).to(agent.device), -agent.action_high, agent.action_high).detach().cpu().tolist()
            action_list.append(target_action)
            s_, r, done, _, _ = env.step(action)
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

       # for l in range(len(ret_lis)):
        #    x_est, P_est = KF(ret_lis[l], x_est, P_est)
         #   ret_lis[l] = x_est
        ret_lis = torch.Tensor(ret_lis).to(agent.device).float()
        agent.lambda_opt.zero_grad()
        Q_ = agent.get_target_val_est(torch.Tensor(state_list).to(agent.device).float(), torch.Tensor(action_list).to(agent.device).float())
        loss = nn.SmoothL1Loss()(Q_, ret_lis.detach())
        loss.backward()
        d = {}

        
        d["lambd_grad_norm"] = torch.nn.utils.clip_grad_norm_([agent.lambd], 100).mean()
        agent.lambda_opt.step()

                
        d["lambd_loss"] = loss.item()
        d["lambd_diff"] = (Q_ - ret_lis.detach()).mean()
        d["return"] = episode_reward
        d["lambd"] = agent.lambd.item()
        wandb.log(step=step, data=d)
        agent.reward_buffer.append(episode_reward)
        mean = np.mean(agent.reward_buffer)
        ys.append(mean)
        
        print("Episode", eps, "Episode Reward", episode_reward, ", Average Reward", 
              mean)
    plt.plot(xs, ys)
    
    
    
    
    
