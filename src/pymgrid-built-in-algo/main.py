import sys
from pymgrid import MicrogridGenerator as mg
import matplotlib.pyplot as plt
import numpy as np
from algorithms import print_architecture, run_benchmark, create_env

nb_microgrids = 3

env = create_env(nb_microgrids)
print_architecture(env)


# run rule base control algorihtm
print("Rule Based Control")
run_benchmark(env, 'rbc', 0.2)
print("-----------------------------------------")

# run model predictive control algorithm
print("Model Predictive Control")
run_benchmark(env, 'mpc', 0.2)
print("-----------------------------------------")

# run sample average approximation algorithm
print("Sample Average Approximation")
run_benchmark(env, 'saa', 0.2)
