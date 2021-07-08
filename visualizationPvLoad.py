from pymgrid import MicrogridGenerator as mg
import numpy as np

m_gen=mg.MicrogridGenerator(nb_microgrid=1)
m_gen.generate_microgrid(verbose=True)
# these values arent change with the time. fix for a grid
print(m_gen.microgrids[0].parameters)

#######################################
#  RUN SIMULATION
#######################################

micro_grid = m_gen.microgrids[0]
# micro_grid.step()

# visualization. this plots the PV and load profiles.
micro_grid.print_load_pv()
