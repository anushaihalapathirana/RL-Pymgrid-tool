from ppotraining import create_agent, training, restore, testing
from pymgrid import MicrogridGenerator as m_gen
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv

number_of_mg = 3
env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
env.generate_microgrid(verbose=False)


mg = env.microgrids[2]
#set cost of co2 for a microgrid. this will set the co2 cost of operating the microgrid at each time step. Default cost of co2 is 0.1
mg.set_cost_co2(0.5)
agent = create_agent(mg)
check_point = training(mg, agent)
new_agent = restore(mg, check_point)
testing(mg, new_agent)

