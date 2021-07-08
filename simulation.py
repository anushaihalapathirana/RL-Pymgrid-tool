
from __future__ import division
  
import csv
import sys
from pymgrid import MicrogridGenerator as mg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

m_gen=mg.MicrogridGenerator(nb_microgrid=1)
m_gen.generate_microgrid()
mg = m_gen.microgrids[0]

mg_data = mg.get_updated_values()

"""
Architecture:
{'PV': 1, 'battery': 1, 'genset': 0, 'grid': 1}

Actions: 
dict_keys(['load', 'pv_consummed', 'pv_curtailed', 'pv', 'battery_charge', 'battery_discharge', 'grid_import', 'grid_export'])

Control dictionnary:
['load', 'pv_consummed', 'pv_curtailed', 'pv', 'battery_charge', 'battery_discharge', 'grid_import', 'grid_export']

Status:  return object of run() method
dict_keys(['load', 'hour', 'pv', 'battery_soc', 'capa_to_charge', 'capa_to_discharge', 'grid_status', 'grid_co2', 'grid_price_import', 'grid_price_export'])

"""

mg.train_test_split() # use train set


 
with open('simulation.csv', 'w') as f:

    # the variable done is used when running a simulation, it will turn to True once you reach the end of load and pv timeseries
    while m_gen.microgrids[0].done == False:
        load = mg_data['load']
        pv = mg_data['pv']
        capa_to_charge = mg_data['capa_to_charge']
        capa_to_dischare = mg_data['capa_to_discharge']
        p_disc = max(0, min(load-pv, capa_to_dischare, mg.battery.p_discharge_max))
        p_char = max(0, min(pv-load, capa_to_charge, mg.battery.p_charge_max))
        
        # this control dictionary has all the power commands that need to be passed to the microgrid in order to operate each generator at each time-step
        control_dict = {'battery_charge': 0,
                            'battery_discharge': p_disc,
                            'grid_import': max(0, load-pv-p_disc),
                            'grid_export':0,
                            'pv_consummed': min(pv, load),
                    }
    
    # run() move forward  one time step, get control dictionary and returns the updated state of the microgrid()
        mg_data = mg.run(control_dict)
        for key in mg_data.keys():
            f.write("%s,%s ,"%(key,mg_data[key]))
        f.write("cost: %s" %mg.get_cost()) # get_cost function returns the cost associated the operation of the last time step.
        f.write(" \n ")   

    if m_gen.microgrids[0].done == True:
        mg.reset(testing=True)

f.close()
    