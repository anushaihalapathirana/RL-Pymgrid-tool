# RL-Pymgrid-tool

This project contains the python scripts of microgrid generation and simulation using Pymgrid tool. (https://github.com/Total-RD/pymgrid)

AI techniques with Pymgrid
* Q-learning
* DQN
* PPO
* Rule-based control algorithm
* Model predictive control
* Sample average approximation

## Code structure
  
``` 
       continuousActionSpace - This folder contains the codes related to continuous action space and continuous state space environment provided by Pymgrid tool.
 
             |
        
             BasicEnvUse - This folder contains the codes that provide examples to how to use the environment provided by the tool
        
             TD3 - codes related to implementation of TD3 algorithm and how to apply TD3 algorithm to pymgrid provided continuous action space and continuous state space environment
             
       discreteActionTuple - How to use discrete action and continuous state space using Pymgrid Tool. Action space is a tuple.
       
       results - results of [simulation_myltiple_microgrids.py] (https://github.com/anushaihalapathirana/RL-Pymgrid-tool/blob/master/simulation_myltiple_microgrids.py) class.
       
       src - This folder contains the RL algorithm implementations for Continuous state and priority list action space environment provided by the Pymgrid tool
       
             |
             
             dqn - Deep Q Network implementation for Pymgrid tool
                   Run maindqn.py file to train, save the model and test
                   Can optimize co2
                   
             ppo - PPO implementation
                   Run main.py file to train, save the model and test
                   Can optimize co2
                   
             pymgrid-built-in-algo - This folder contains the built in RL algorithms in Pymgrid tool
                                     run main.py file - this will run Rule Based Control, Model Predictive Control, and Sample Average Approximation at once and save the model. 
                                     
             q-learning - Q learning implementation for pymgrid tool
                  
                  |
                  
                  q-learning.py - Q learning with default cost
                  
                  q-learning-co2-cost-opt.py - Q learning with co2  cost optimization
                  
                  q-learning-action-setup.py - How to set up our own action space and use Q learning
                            
  ```
  
  You can found the full documentation from below links
  
  1. https://github.com/anushaihalapathirana/RL-Pymgrid-tool/blob/master/Pymgrid-tool-basics.pdf
  2. https://github.com/anushaihalapathirana/RL-Pymgrid-tool/blob/master/Pymgrid-Tool-Implementation.pdf
 

