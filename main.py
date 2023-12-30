

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:04:16 2022

@author: guanfei1
"""

import gymnasium as gym
import numpy as np
from TD3 import *
import matplotlib.pyplot as plt
import wandb
import time
if __name__ == '__main__':
    env = gym.make('Ant-v2')
    project_name = 'TD3'
    wandb.init(project=project_name, entity='fguan06', settings=wandb.Settings(start_method="thread"))
    wandb.run.name = "TD3-Baseline-Ant-v2"

    agent = TD3(env)
    ys = []
    eps = 2000
    ret = 0.

    xs = list(range(eps))
    step = 0
    print("Replay Buffer Initialized")
    trunc = False
    b = 0
    eps = 0
    while not step > 1000000:
        done = False
        episode_reward = 0
        s = env.reset()[0]
        eps += 1
        x = time.time()
        while not (trunc or done):
            action = agent.act(s)
            s_, r, done, trunc, _ = env.step(action)
            agent.replay_buffer.append((s, action, s_, r, done))
            
            s = s_
            episode_reward += r
            step += 1
            if len(agent.replay_buffer) > agent.batch_size :
                eva = False
                agent.train()

                
            if done:
                break 
                

        d = {}
        d["return"] = episode_reward
        wandb.log(step=step, data=d)
        agent.reward_buffer.append(episode_reward)
        mean = np.mean(agent.reward_buffer)
        ys.append(mean)
        
        print("Episode", eps, "Episode Reward", episode_reward, ", Average Reward", 
              mean)
    plt.plot(xs, ys)    
    
