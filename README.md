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
       
       src - 
             
  ```

