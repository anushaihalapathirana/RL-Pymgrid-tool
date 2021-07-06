import sys
import subprocess, re, os, sys

from pymgrid import MicrogridGenerator as m_gen
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import shutil

import ray
from ray.rllib.agents import ppo


from pymgrid.Environments.pymgrid_cspla import MicroGridEnv

class ppo:

    number_of_mg = 1
    env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
    env.generate_microgrid(verbose=False)

    ray.shutdown()
    ray.init(num_cpus=2, num_gpus=1) 

    # checkpoint configuration
    CHECKPOINT_ROOT = "tmp/ppo"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = os.getenv("HOME") + "/ray_results/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    def get_config(mg):
        # training a PPO on microgrid, use config to change parameters for the training
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_workers"] = 1
        config["model"]["fcnet_hiddens"] = [100, 100]
        config["num_gpus"] = 1
        config["env_config"]={'microgrid':mg}
        config["batch_mode"] = "complete_episodes"
        return config

    def create_agent(mg):
        config = get_config(mg)
        agent = ppo.PPOTrainer(env=MicroGridEnv, config=config)
        return agent

    def training(mg, agent):
        nb_epoch = 5 # length of the training process
        results = []
        episode_data = []
        episode_json = []
        checkpoint_path=''
        reward_mean = float('-inf')

        for i in range(nb_epoch):
            result = agent.train()
            results.append(result)

            # add checkpoint for best max reward 
            if result['episode_reward_max'] > reward_mean:
                checkpoint_path = agent.save()
                reward_mean = result['episode_reward_max']

            episode = {'n': i,
                    'episode_reward_min': result['episode_reward_min'],
                    'episode_reward_mean': result['episode_reward_mean'],
                    'episode_reward_max': result['episode_reward_max'],
                    'episode_len_mean': result['episode_len_mean']
                    }
            episode_data.append(episode)
            print(f'{i:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.1f}/{result["episode_reward_mean"]:8.1f}/{result["episode_reward_max"]:8.1f}')
        return checkpoint_path


    def restore(checkpoint_path):
        # restore the policy
        config = get_config(mg)
        trained_config = config.copy()
        test_agent = ppo.PPOTrainer(env=MicroGridEnv, config=trained_config)
        test_agent.restore(checkpoint_path)
        return test_agent


    def testing(mg, agent):
        # check the performance of the policy on the testing set
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

        mg.print_cumsum_cost()
        mg.print_co2()
            
        plt.plot(rewards)
        plt.ylabel('cost')
        plt.xlabel('episodes')
        plt.show()

        print(f"Testing reward for microgrid {episode_reward}")

# mg = env.microgrids[0]
# agent = create_agent(mg)
# check_point = training(mg, agent)
# new_agent = restore(check_point)
# testing(mg, new_agent)