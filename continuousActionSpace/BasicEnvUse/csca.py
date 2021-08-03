
# How to access the continuous state and continuous action space in pymgrid
from pymgrid import MicrogridGenerator as m_gen
import numpy as np

from pymgrid.Environments.pymgrid_csca import ContinuousMicrogridEnv

number_of_mg = 1
env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
env.generate_microgrid(verbose=False)
mg = env.microgrids[0]

env = ContinuousMicrogridEnv(mg)
print("")
print('state space: ', env.observation_space)
print('action space: ', env.action_space)
print("")

