import numpy as np
import os, sys
import gym
from agent import Agent


def testTD3(env):
    
    agent = Agent(alpha=0.001, beta=0.001, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=100, layer1_size=400, layer2_size=300,
                n_actions=env.action_space.shape[0])
   
    n_games = 10
    
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        state = env.reset()
        score = 0
        for j in range(8759):
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, new_state, done)
            agent.load_models()
            score += reward
            state = new_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
    
    mean_reward = np.mean(score_history)
    std_reward = np.std(score_history)
    print('mean cost is  %.2f' % mean_reward, 'std_cost %.3f' % std_reward)

