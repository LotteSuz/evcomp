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
n_pop = 20
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

def order_to_fitness(fitness,pop):
    fitness,pop = zip(*sorted(zip(fitness,pop),key = lambda fit:fit[0],reverse = True))
    return list(fitness),list(pop)

def make_children(pop, fitness, n_pop):
    # order list from high to low fitness ;
    kids = []
    fitness,pop = order_to_fitness(fitness,pop)
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
       print(ind1,ind2)
    return kids

def mutate(kids):
    to_mutate = int(mutation*len(kids))
    mutated_kids = []
    for i in range(to_mutate):
        ind = np.random.randint(len(kids)-1)
        while ind in mutated_kids:
            ind = np.random.randint(len(kids)-1)
        toflip = np.random.randint(n_vars)
        print('fitnes before mutation : ',evaluate([kids[ind]]))
        kids[ind][toflip] = np.random.uniform(-1,1)
        print('fitness after mutation : ',evaluate([kids[ind]]))
        mutated_kids.append(ind)
    return kids

def kill(pop,fitness,kids):
    fitness, pop = order_to_fitness(fitness,pop)
    kids_fitness = evaluate(kids)
    kids_fitness,kids = order_to_fitness(kids_fitness,kids)
    print('list of population fitnes ;',fitness)
    print('list of kids fitness ; ',kids_fitness)
    index = 0
    for i in range(len(kids)):
        for j in range(index,len(pop)):
            if kids_fitness[i] > fitness[j]:
                pop[j] = kids[i]
                fitness[j] = kids_fitness[i]
                index = j
                break

    return pop,fitness


# instantiate population
pop = np.random.uniform(dom_l, dom_u, (n_pop, n_vars))
fitness = evaluate(pop)
kids = make_children(pop, fitness, n_pop)
pop,fitness = kill(pop,fitness,kids)
kids = mutate(kids)

# correlated mutations (CMA-ES)
