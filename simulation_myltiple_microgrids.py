
from __future__ import division
  
import csv
import sys
from pymgrid import MicrogridGenerator as mg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

number_of_mg = 3
m_gen=mg.MicrogridGenerator(nb_microgrid=number_of_mg)
m_gen.generate_microgrid()


for i in range(number_of_mg):

    f = open("/home/kali/Documents/intern/openaigym/myPymgridTests/results/test%x.csv"%i, 'w')
    mg = m_gen.microgrids[i]
    print("Microgrid {} architecture: {}".format(int(i), str(m_gen.microgrids[i].architecture)))
    mg_data = mg.get_updated_values()

    # use only 67% of the data 
    # mg.train_test_split()

    # the variable done is used when running a simulation, it will turn to True once you reach the end of load and pv timeseries
    while mg.done == False:
        load = mg_data['load']
        pv = mg_data['pv']
        capa_to_charge = mg_data['capa_to_charge']
        capa_to_dischare = mg_data['capa_to_discharge']
        p_disc = max(0, min(load-pv, capa_to_dischare, mg.battery.p_discharge_max))
        p_char = max(0, min(pv-load, capa_to_charge, mg.battery.p_charge_max))
            
        # this control dictionary has all the power commands that need to be passed to the microgrid in order to operate each generator at each time-step
        control_dict = {'battery_charge': p_char,
                            'battery_discharge': p_disc,
                            'grid_import': max(0, load-pv-p_disc),
                            'grid_export':0,
                            'pv_consummed': min(pv, load),
                    }
        
        # run() move forward  one time step, get control dictionary and returns the updated state of the microgrid()
        mg_data = mg.run(control_dict)
        
        for key in mg_data.keys():
            f.write("%s,%s ,"%(key,mg_data[key]))
        f.write("cost: %s" %mg.get_cost())  # Function that returns the cost associated the operation of the last time step
        f.write(" \n ")  
    f.close()  

    mg.print_actual_production()

    # plot the co2 emissions associated to the operation of the time step. no co2 emmission if grid is not available
    mg.print_co2()

    # plot the cost of operating the microgrid at each time step. 
    mg.print_cumsum_cost()

    if mg.done == True:
        print("-------------------")
        mg.reset()

    
    