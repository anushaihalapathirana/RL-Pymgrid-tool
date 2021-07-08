from pymgrid import MicrogridGenerator as mg
import numpy as np

m_gen=mg.MicrogridGenerator(nb_microgrid=5)
m_gen.generate_microgrid()

# m_gen.microgrids[0].get_control_inf()


# access battery details
print(m_gen.microgrids[0].battery.soc)

# access genset data
print(m_gen.microgrids[1].genset.p_max)

# access genset data
print(m_gen.microgrids[0].grid.status)

#######################################
#  RUN SIMULATION
#######################################

micro_grid = m_gen.microgrids[0]
print("===========================")
micro_grid.print_info()
print("===========================")
micro_grid.print_control_info()
print("===========================")
print(micro_grid.load)
print("===========================")

# visualization. this plots the PV and load profiles.
print(micro_grid.print_load_pv())