
# How to access the continuous state and discrete action space in pymgrid
from pymgrid import MicrogridGenerator as m_gen
import numpy as np

from pymgrid.Environments.pymgrid_csda import MicroGridEnv

number_of_mg = 1
env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
env.generate_microgrid(verbose=False)
mg = env.microgrids[0]

env = MicroGridEnv({'microgrid': mg})
print("")
print('state space: ', env.observation_space)
print('action space: ', env.action_space)
print("")
