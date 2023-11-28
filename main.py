#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:04:16 2022

@author: guanfei1
"""

import gym
import numpy as np
from gym import wrappers
from TD3 import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v2')
    agent = TD3(env)
    ys = []
    eps = 2000
    xs = list(range(eps))
    print("Replay Buffer Initialized")
    for j in range(eps):
        done = False
        episode_reward = 0
        s = env.reset()
        while not done:
            action = agent.act(s)
            s_, r, done, _ = env.step(action)
            agent.replay_buffer.append((s, action, s_, r, done))
            s = s_
            episode_reward += r
            env.render()
            if len(agent.replay_buffer) > agent.batch_size:
                agent.train()
        agent.reward_buffer.append(episode_reward)
        mean = np.mean(agent.reward_buffer)
        ys.append(mean)
        
        print("Episode", j, "Episode Reward", episode_reward, ", Average Reward", 
              mean)
    plt.plot(xs, ys)
    
    
