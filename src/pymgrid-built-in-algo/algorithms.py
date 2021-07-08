import sys
from pymgrid import MicrogridGenerator as mg
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv
import matplotlib.pyplot as plt
import numpy as np

def create_env(nb_microgrids):
    env = mg.MicrogridGenerator(nb_microgrid=nb_microgrids)
    env.generate_microgrid(verbose = False)
    return env

def print_architecture(env):
    print("--------------------------------------------")

    # print the architecture of these 3 microgrids
    for i in range(env.nb_microgrids):
        print("Microgrid {} architecture: {}".format(int(i), str(env.microgrids[i].architecture)))

    print("--------------------------------------------")


def run_benchmark(env, algorithm, co2_cost):
    for i in range(env.nb_microgrids):
        print("Microgrid", i,":")
        mg = env.microgrids[i]
        mg.set_cost_co2(co2_cost)
        # mg.train_test_split()
        mg.benchmarks.run_benchmarks(algo = algorithm) # this run control algorithms. Currently supports MPC and rule-based control
        mg.benchmarks.describe_benchmarks()


