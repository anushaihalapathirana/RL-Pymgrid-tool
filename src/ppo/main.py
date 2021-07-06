from ppotraining import create_agent, training, restore, testing
from pymgrid import MicrogridGenerator as m_gen
from pymgrid.Environments.pymgrid_cspla import MicroGridEnv

number_of_mg = 1
env=m_gen.MicrogridGenerator(nb_microgrid=number_of_mg)
env.generate_microgrid(verbose=False)


mg = env.microgrids[0]
agent = create_agent(mg)
check_point = training(mg, agent)
new_agent = restore(mg, check_point)
testing(mg, new_agent)

