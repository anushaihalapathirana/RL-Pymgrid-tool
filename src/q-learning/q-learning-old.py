import sys
from pymgrid import MicrogridGenerator as mg
import matplotlib.pyplot as plt
import numpy as np
import os
import time

"""
    Q-learning implementation for pymgrid tool.
"""
number_of_mg = 3
env=mg.MicrogridGenerator(nb_microgrid=number_of_mg)
env.generate_microgrid(verbose=False)
mg0 = env.microgrids[0]

# print the architecture
for i in range(env.nb_microgrids):
    print("Microgrid {} architecture: {}".format(int(i), str(env.microgrids[i].architecture)))

"""
    actions that agent can take

    action 0: battery_charge
    action 1: battery_discharge
    action 2: grid_import
    action 3: grid_export

    return dictionary. because run() needs dictionary as a parameter
"""

def actions_agent(microgrid, action):
    
    pv = microgrid.pv
    load = microgrid.load

    net_load = load - pv

    capacity_to_charge = microgrid.battery.capa_to_charge # amount of energy that a battery can charge before being full
    capacity_to_discharge = microgrid.battery.capa_to_discharge # the amount of energy available that a battery can discharge before being empty

    max_battery_charging_rate = microgrid.battery.p_charge_max # Value representing the maximum charging rate of the battery (kW)
    max_battery_discharging_rate = microgrid.battery.p_discharge_max # Value representing the maximum charging rate of the battery (kW)

    charge = max(0, min(-net_load, capacity_to_charge, max_battery_charging_rate))
    discharge = max(0, min(net_load, capacity_to_discharge, max_battery_discharging_rate))

    pv_consumption = min(pv, load)
    grid_export_amount = pv - pv_consumption - charge
    grid_import_amount = load - pv_consumption - discharge
    abs_net_load = abs(net_load)

    if action == 0: # battery charge action

        if charge > 0:
            control_dict = {'battery_charge': charge, # Battery capacity to charge
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': max(0, grid_export_amount),
                            'pv_consummed': pv_consumption,
                        }
        else: # charge is 0
            control_dict = {'battery_charge': net_load, #??? 
                            'battery_discharge': 0,
                            'grid_import': 0,
                            'grid_export': max(0, grid_export_amount),
                            'pv_consummed': pv_consumption,
                        }

    elif action == 1: # battery discharge
        if discharge > 0:
            control_dict = {'battery_charge': 0,
                            'battery_discharge': discharge,
                            'grid_import': max(0, grid_import_amount),
                            'grid_export': 0,
                            'pv_consummed': pv_consumption,
                        }
        else:
            control_dict = {'battery_charge': 0,
                            'battery_discharge': net_load,
                            'grid_import': max(0, grid_import_amount),
                            'grid_export': 0,
                            'pv_consummed': pv_consumption,
                        }

    elif action == 2: # grid import
        control_dict = {'battery_charge': 0,
                        'battery_discharge': 0,
                        'grid_import': abs_net_load,
                        'grid_export': 0,
                        'pv_consummed': pv_consumption,
                    }

    elif action == 3: # grid export
        control_dict = {'battery_charge': 0,
                        'battery_discharge': 0,
                        'grid_import': 0,
                        'grid_export': abs_net_load,
                        'pv_consummed': pv_consumption,
                    }
    return control_dict


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
def get_max(q_table):
    max_key = None
    max_value = float('-inf')

    for key, value in q_table.items():
        if value > max_value:
            max_key = key
            max_value = value
    return max_key, max_value


# update ecploration rate
def update_exploration_rate(exploration_rate, episode):
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    return exploration_rate

def get_action_name(action):
    action_name = ""
    if action == 0:
        action_name = "battery charge"
    elif action == 1:
        action_name = "battery discharge"
    elif action == 2:
        action_name = "grid import"
    elif action == 3:
        action_name = "grid_export"
    else:
        action_name = ""
    return action_name


def training(microgrid, horizon):

    number_of_actions = 4
    q_table = initialize_q_table(microgrid, number_of_actions)
    number_of_states = len(q_table)
    number_of_episodes = 100

    learning_rate = 0.1
    discount_rate = 0.9
    exploration_rate = 1

    cost_of_episodes = []

    for episode in range(number_of_episodes+1):
        current_episode_cost = 0
        microgrid.reset()
        net_load = round(microgrid.load - microgrid.pv)
        state_of_capacity = round(microgrid.battery.soc, 1)

        state = (net_load, state_of_capacity)
        action = get_max(q_table[state])[0]
        action = exploration_greedy_function(action, exploration_rate, number_of_actions)

        for step in range(horizon):

            action_to_run = actions_agent(microgrid, action)
            status = microgrid.run(action_to_run)
            reward = -microgrid.get_cost()
            current_episode_cost += microgrid.get_cost()

            net_load = round(microgrid.load - microgrid.pv)
            state_of_capacity = round(microgrid.battery.soc, 1)

            new_state = (net_load, state_of_capacity)
            new_action = get_max(q_table[new_state])[0]

            if step == horizon -1:
                q_table[state][action] = q_table[state][action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * q_table[state][action])
            else:
                target = reward + discount_rate * q_table[new_state][new_action]
                q_table[state][action] =  q_table[state][action] * (1 - learning_rate) + learning_rate * target

            state = new_state
            action = new_action

        exploration_rate = update_exploration_rate(exploration_rate, episode)

    return q_table


def testing(microgrid, q_table, horizon):
    microgrid.reset()
    net_load = round(microgrid.load - microgrid.pv)
    soc = round(microgrid.battery.soc, 1)
    state = (net_load, soc)
    action = get_max(q_table[state])[0]

    total_cost = 0

    print("t -     STATE  -  ACTION -   COST  -  TOTAL COST  -  CO2 EMISSION")
    print("================================ \n")
        
    for step in range(horizon):
        name_of_action = get_action_name(action)
        action_dictionary = actions_agent(microgrid, action)
        status = microgrid.run(action_dictionary)
        cost = microgrid.get_cost()

        co2 = microgrid.get_co2()

        total_cost += cost

        print(step,"-",(int(net_load),soc),name_of_action, round(cost,1), "€", round(total_cost,1), "€", round(co2,1))

        net_load = round(microgrid.load - microgrid.pv)
        soc = round(microgrid.battery.soc, 1)
        
        new_state = (net_load, soc)
        new_action = get_max(q_table[new_state])[0]

        state = new_state
        action = new_action



print("\n========== TRAINING START =============")
q_table = training(mg0,24)
print("========== TRAINING END =============\n")

testing(mg0, q_table, 24)