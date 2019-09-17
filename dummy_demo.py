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


#env.play()

# set time marker
ini = time.time()

# define parameters
n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
n_pop = 10
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
    new_agent = np.concatenate((agent_x[:ind], agent_y[ind:]), axis=None)
    return new_agent

def make_children(pop, fitness, n_pop):
    # order list from high to low fitness ;
    kids = []
    fitness,pop = zip(*sorted(zip(fitness,pop),key = lambda fit:fit[0],reverse = True))
    chance = [(float(i)-fitness[-1]) / (fitness[0] - fitness[-1]) for i in fitness]
    cumchance = np.cumsum(chance)
    n_kids = int(0.25 * n_pop)
    for i in range(n_kids):
       first = random.uniform(0, cumchance[-1])
       second = random.uniform(0, cumchance[-1])
       for i in range(len(cumchance)):
           if first < cumchance[i]:
               ind1 = i
               break
       for i in range(len(cumchance)):
           if second < cumchance[i]:
               ind2 = i
               break
       kids.append(crossover(pop[ind1], pop[ind2]))
    return kids

# instantiate population
pop = np.random.uniform(dom_l, dom_u, (n_pop, n_vars))
fitness = evaluate(pop)
kids = make_children(pop, fitness, n_pop)

# correlated mutations
