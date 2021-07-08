import sys
from pymgrid import MicrogridGenerator as mg
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv
import matplotlib.pyplot as plt
import numpy as np
import os
import time

env = mg.MicrogridGenerator(nb_microgrid=3)
env.generate_microgrid(verbose = False)

print("--------------------------------------------")

# print the architecture of these 2 microgrids
for i in range(env.nb_microgrids):
    print("Microgrid {} architecture: {}".format(int(i), str(env.microgrids[i].architecture)))

print("--------------------------------------------")

for i in range(env.nb_microgrids):
    print("Microgrid", i,":")
    mg = env.microgrids[i]
    mg.set_cost_co2(0.2)
    mg.train_test_split()
    mg.benchmarks.run_benchmarks(algo = "rbc") # this run control algorithms. Currently supports MPC and rule-based control
    mg.benchmarks.describe_benchmarks()
