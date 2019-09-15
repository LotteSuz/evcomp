################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

print('poop')

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment

print('poop')

# import libraries
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from controller_specialist_demo import player_controller

print('poop')

experiment_name = 'dummy_demo_lau'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

def init_pop(pop_size, env):
    # Function that initializes population in the same way as done in example (optimization_specialis.py)
    n_hidden = 10
    agent_variables = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
    # Actual population: 
    pop = np.random.uniform(-1, 1, (pop_size, agent_variables))
    return pop


# initializes environment with ai player using random controller, playing against static enemy
#env = Environment(experiment_name=experiment_name,solutions = None)
pop = init_pop(10,env)
print(len(pop))

#env.play()

# set time marker
# ini = time.time()

# define parameters
n_hidden = 10

## n_vars = 265 in opt_spec_demo.py =
# run simulation
# normalize (to avoid negative probabilities of survival)
# evaluation (calculates performance of every agent)
# tournament (pick 2 players, keep the fittest aka natural selection)
# limits ?
# crossover (recombination & mutation)
# kill worst genomes > replace with new best/random solutions
