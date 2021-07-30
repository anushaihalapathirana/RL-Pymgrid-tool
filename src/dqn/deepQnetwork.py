import sys
from pymgrid import MicrogridGenerator as m_gen
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import shutil

import ray
from ray.rllib.agents import dqn

number_of_mg = 1
env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
env.generate_microgrid(verbose=False)
mg = env.microgrids[0]

ray.shutdown()
ray.init() 

# checkpoint configuration
CHECKPOINT_ROOT = "tmp/dqn"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

def get_config(mg):
    # training a DQN on microgrid, change parameters for the training process using config
    config = dqn.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["env_config"]={'microgrid':mg}
    config["batch_mode"] = "complete_episodes"
    return config

def create_dqn_agent(mg):
    config = get_config(mg)
    agent = dqn.DQNTrainer(env=MicroGridEnv, config=config)
    return agent 

def training_dqn(mg, agent):
    nb_epoch = 100 # length of the training process
    results = []
    checkpoint_path=''
    reward_mean = float('inf')
    
    for i in range(nb_epoch):
        result = agent.train()
        results.append(result)
        
        # add checkpoint for best max reward 
        if result['episode_reward_mean'] < reward_mean:
            checkpoint_path = agent.save()
            reward_mean = result['episode_reward_mean']

        episode = {'n': i,
                'episode_reward_min': result['episode_reward_min'],
                'episode_reward_mean': result['episode_reward_mean'],
                'episode_reward_max': result['episode_reward_max'],
                'episode_len_mean': result['episode_len_mean']
                }
        print(f'{i:3d}: Min | Mean | Max reward: {result["episode_reward_min"]:8.1f} | {result["episode_reward_mean"]:8.1f} | {result["episode_reward_max"]:8.1f}')
    return checkpoint_path


def restore(mg, checkpoint_path):
    config = get_config(mg)
    # restore the checkpoint
    trained_config = config.copy()
    test_agent = dqn.DQNTrainer(env=MicroGridEnv, config=trained_config)
    test_agent.restore(checkpoint_path)
    return test_agent


def testing_dqn(mg, agent):
    # check the performance on the testing set
    env = MicroGridEnv({'microgrid':mg})
    episode_reward = 0
    done = False
    obs = env.reset(testing=True)
    rewards = []
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        rewards.append(-reward)

    # mg.print_cumsum_cost()

    mg.print_cumsum_co2_cost()

    # mg.print_co2()
        
    # plt.plot(rewards)
    # plt.ylabel('cost')
    # plt.xlabel('episodes')
    # plt.show()
    print(f"Testing reward for microgrid {-episode_reward}")
    
