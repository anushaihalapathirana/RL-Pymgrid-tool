from pymgrid import MicrogridGenerator as mg
import numpy as np

"""
Microgrid list generation
"""
m_gen=mg.MicrogridGenerator(nb_microgrid=2)
m_gen.generate_microgrid()

# Choose microgrid 0 and print main information about the microgrid
mg = m_gen.microgrids[1]

"""
Microgrid 0 architecture has no grid.
so in main informations grid related data is missing
"""
print("\n ============     MICROGRID MAIN INFORMATION    ============== \n")
mg.print_info()


#access microgrid data
print("\n ============     ACCESS MICROGRID DATA     ==============")
battery_cap = mg.battery.capacity
battery_efficiency = mg.battery.efficiency
grid_status = mg.grid.status
print("Battery capacity : ", battery_cap)
print("Battery efficiency : ", battery_efficiency)
print("Grid status : ", grid_status)


# control_dict fields that used to control the microgrid
print("\n ============     CONTROL DICTIONARY FIELDS     ==============")
mg.print_control_info()

# control_dict data that used to control the microgrid
print("\n =========    CONTROL DICTIONARY DATA     ===========")
mg_data = mg.get_updated_values()
print(mg_data)  
