import gym
import sys
from traintd3 import trainTD3
from testtd3 import testTD3

from pymgrid import MicrogridGenerator as m_gen
import numpy as np

from pymgrid.Environments.pymgrid_csca import ContinuousMicrogridEnv

if __name__ == '__main__':
    number_of_mg = 1
    env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
    env.generate_microgrid(verbose=False)
    mg = env.microgrids[0]

    env = ContinuousMicrogridEnv(mg)

    trainTD3(env)
    print("*******  Training Done  *************")
    testTD3(env)