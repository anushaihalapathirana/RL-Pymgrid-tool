from pymgrid import MicrogridGenerator as mg
import numpy as np

# m_gen=mg.MicrogridGenerator(nb_microgrid=10,random_seed=7)
# m_gen.generate_microgrid()
# mg = m_gen.microgrids

# mg_data = mg.get_updated_values()
# print(mg_data)


# # mg.print_info()
# # mg.print_control_info()

# a = m_gen.load('pymgrid25')

from pymgrid.Environments.pymgrid_csca import ContinuousMicrogridEnv
from pymgrid import MicrogridGenerator as m_gen

#these line will create a list of microgrid
env = m_gen.MicrogridGenerator(nb_microgrid=25)
pymgrid25 = env.load('pymgrid25')
mg = pymgrid25.microgrids

for m in mg:
    print(m.architecture)

env = ContinuousMicrogridEnv(mg[0])
print(env.observation_space)
print(env.action_space)
