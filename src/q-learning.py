import sys
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv
from pymgrid import MicrogridGenerator as mg
import matplotlib.pyplot as plt
import numpy as np
import os
import time

"""
    Q-learning implementation for pymgrid tool.
"""
number_of_mg = 3
horizon = 48
env=mg.MicrogridGenerator(nb_microgrid=number_of_mg, random_seed = 42)
env.generate_microgrid(verbose=False)
mg0 = env.microgrids[0]
# set horizon. default value is 24
mg0.set_horizon(horizon)

#set cost of co2 for a microgrid. this will set the co2 cost of operating the microgrid at each time step. Default cost of co2 is 0.1
mg0.set_cost_co2(0.5)

# print the architecture
for i in range(env.nb_microgrids):
    print("Microgrid {} architecture: {}".format(int(i), str(env.microgrids[i].architecture)))

# to use provided environment
env = MicroGridEnv({'microgrid':mg0})

# initialize q table as dictionary -> {(247, 0.2): {0: 0, 1: 0, 2: 0},...} (forcasted netload, soc)
def initialize_q_table(microgrid, number_of_actions):
    # gives the load and pv forecasted values for the next horizon (24 hours)
    net_load = microgrid.forecast_load() - microgrid.forecast_pv() # array with 24 values

    states = []
    q_table = {}

    # create discrete states using battery state of charge and forcasted net load
    for i in range(int(net_load.min()-1),int(net_load.max()+2)):
        for j in np.arange(round(mg0.battery.soc_min,1),round(mg0.battery.soc_max+0.1,1),0.1):
            j = round(j,1)
            states.append((i,j)) 

    for state in states:
        q_table[state] = {}
        for action in range(number_of_actions):
            q_table[state][action] = 0

    return q_table
         

# select action based on the exploration value
def exploration_greedy_function(action, exploration_rate, number_of_actions):
    exploration_rate_threshold = np.random.uniform(0,1)
    if exploration_rate_threshold < exploration_rate:
        return action
    else:
        return np.random.choice(number_of_actions)


# get the maximum value of q_table
def get_max_q_value(q_table):
    max_key = None
    max_value = float('-inf')

    for key, value in q_table.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key, max_value


def get_action_name(action):
    action_name = ""
    if action == 0 or action == 5:
        action_name = "battery charge"
    elif action == 1:
        action_name = "battery discharge"
    elif action == 2:
        action_name = "grid import"
    elif action == 3:
        action_name = "grid_export"
    elif action == 4:
        action_name = "genset"
    elif action == 6:
        action_name = "genset and battery discharge"
    else:
        action_name = "unknown action"
    return action_name


# update the exploration rate according to the epsilon greedy algorithm
def update_exploration_rate(exploration_rate, episode):
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01
    
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    return exploration_rate

def training(microgrid, horizon):

    number_of_actions = env.action_space.n
    q_table = initialize_q_table(microgrid, number_of_actions)
    number_of_states = len(q_table)

    number_of_episodes = 1000

    learning_rate = 0.01
    discount_rate = 0.9
    exploration_rate = 1

    cost_of_episodes = []

    for episode in range(number_of_episodes+1):
        current_episode_cost = 0
        microgrid.reset()
        net_load = round(microgrid.load - microgrid.pv)
        state_of_capacity = round(microgrid.battery.soc, 1)

        state = (net_load, state_of_capacity)
        action = get_max_q_value(q_table[state])[0]
        action = exploration_greedy_function(action, exploration_rate, number_of_actions)

        for step in range(horizon):

            next_action = env.get_action(action)
            status = microgrid.run(next_action)

            reward = -microgrid.get_cost()
            current_episode_cost += microgrid.get_cost()

            # net_load = round(microgrid.load - microgrid.pv)
            # state_of_capacity = round(microgrid.battery.soc, 1)

            net_load = round(status['load']  - status['pv'])
            state_of_capacity = round(status['battery_soc'], 1)

            new_state = (net_load, state_of_capacity)
            new_action = get_max_q_value(q_table[new_state])[0]

            # update the q table
            if step == horizon -1:
                q_table[state][action] = q_table[state][action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * q_table[state][action])
            else:
                target = reward + discount_rate * q_table[new_state][new_action]
                q_table[state][action] =  q_table[state][action] * (1 - learning_rate) + learning_rate * target

            state = new_state
            action = new_action

        exploration_rate = update_exploration_rate(exploration_rate, episode)

        cost_of_episodes.append(current_episode_cost)

    return q_table


def testing(microgrid, q_table, horizon):
    microgrid.reset()
    net_load = round(microgrid.load - microgrid.pv)
    soc = round(microgrid.battery.soc, 1)
    state = (net_load, soc)
    action = get_max_q_value(q_table[state])[0]

    total_cost = 0
    cost_of_horizon = []

    print("t - STATE(net load, soc)  -  ACTION -   COST  -  TOTAL COST  -  CO2 EMISSION")
    print("================================ \n")
        
    for step in range(horizon):
        name_of_action = get_action_name(action)
        action_dictionary = env.get_action(action)
        status = microgrid.run(action_dictionary)

        cost = microgrid.get_cost()
        co2 = microgrid.get_co2()

        total_cost += cost
        # print(status['grid_co2'])

        print(step,"-",(int(net_load),soc),name_of_action, round(cost,1), "€", round(total_cost,1), "€", round(co2,1))

        # net_load = round(microgrid.load - microgrid.pv)
        # soc = round(microgrid.battery.soc, 1)
        
        net_load = round(status['load']  - status['pv'])
        soc = round(status['battery_soc'], 1)

        new_state = (net_load, soc)
        new_action = get_max_q_value(q_table[new_state])[0]

        state = new_state
        action = new_action
        cost_of_horizon.append(total_cost)
    
    print('\n Total operating cost of the microgrid is : ', round(total_cost, 1), " € \n")
    
    plt.plot(cost_of_horizon)
    plt.title('Total operating cost')
    plt.ylabel('cost')
    plt.xlabel('horizon (in hour)')
    plt.show()



print("\n TRAINING... \n")
q_table = training(mg0,horizon)
print(" TRAINING END \n")

testing(mg0, q_table, horizon)