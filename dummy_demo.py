################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller


# import libraries
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name='saturday',
                  enemies=[1],
                  player_controller=player_controller(),
                  level=1,
                  speed="fastest")

# THIS RUNS THE VISUALIZATION
#env.play()

# set time marker
ini = time.time()

# define parameters
n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
n_pop = 2
n_gen = 30
mutation = 0.2
last_best = 0

def evaluate(pop):
    fitness = []
    for agent in pop:
        f,p,e,t = env.play(pcont=agent)
        fitness.append(f)
    print(f"best fitness = {max(fitness)}")
    return fitness

def crossover(agent_x, agent_y):
    ind = np.random.randint(1,n_vars-2)
    new_agent = [agent_x[1:ind] + agent_y[ind+1:len(agent_y)]]
    #new_agent.append(agent_x[1:ind])
    #new_agent.append(agent_y[ind+1:len(agent_y)])
    #agent_x[1:ind] + agent_y[ind+1:len(agent_y)]
    print(f"newagent = {len(new_agent)}")
    return new_agent




# instantiate population
pop = np.random.uniform(dom_l, dom_u, (n_pop, n_vars))
evaluate(pop)
crossover(random.choice(pop), random.choice(pop))


# normalize (to avoid negative probabilities of survival)
# evaluation (calculates performance of every agent)
# tournament (pick 2 players, keep the fittest aka natural selection)
# limits ?
# crossover (recombination & mutation)
# kill worst genomes > replace with new best/random solutions
