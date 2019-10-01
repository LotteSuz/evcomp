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
n_pop = 12
n_gen = 3
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
    new_agent_first = np.concatenate((agent_x[:ind], agent_y[ind:]), axis=None)
    new_agent_second = np.concatenate((agent_x[ind:], agent_y[:ind]), axis=None)
    return new_agent_first, new_agent_second

def order_to_fitness(fitness,pop):
    fitness,pop = zip(*sorted(zip(fitness,pop),key = lambda fit:fit[0],reverse = True))
    return list(fitness),list(pop)


def make_children(pop, fitness, n_pop):
    # order list from high to low fitness ;
    kids = []
    family = []
    fitness,pop = order_to_fitness(fitness,pop)
    chance = [(float(i)-fitness[-1]) / (fitness[0] - fitness[-1]) for i in fitness]
    cumchance = np.cumsum(chance)
    n_kids = int(0.25 * n_pop)
    c0 = 0
    c1 = 1
    for i in range(n_kids):
        while True:
            first = random.uniform(0, cumchance[-1])
            second = random.uniform(0, cumchance[-1])
            for j in range(len(cumchance)):
                if first < cumchance[j]:
                    ind1 = j
                    break
            for k in range(len(cumchance)):
                if second < cumchance[k]:
                    ind2 = k
                    break
            if (ind1 != ind2):
                break
        first, second = crossover(pop[ind1], pop[ind2])
        kids.append(first)
        kids.append(second)
        #family.append(dict({"offspring1": [c0, first], "offspring2": [c1, second],
        #"parent1": [ind1, pop[ind1]], "parent2": [ind2, pop[ind2]]}))
        family.append([c0,c1,ind1,ind2])
        c0 += 2
        c1 += 2
    return kids, family

def mutate(kids):
    to_mutate = int(mutation*len(kids))
    mutated_kids = []
    for i in range(to_mutate):
        ind = np.random.randint(len(kids)-1)
        while ind in mutated_kids:
            ind = np.random.randint(len(kids)-1)
        mutated_kids.append(ind)
        for i in range(50):
            toflip = np.random.randint(n_vars)
            kids[ind][toflip] = np.random.uniform(-1,1)
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

def diversity(kids, pop, family):
    # get fitnesses of all family members
    for fam in family:
        performance = evaluate([kids[fam[0]],kids[fam[1]],pop[fam[2]],pop[fam[3]]])
        # compute intercompetition distances
        left = abs(performance[0] - performance[2]) + abs(performance[1] - performance[3])
        right = abs(performance[0] - performance[3]) + abs(performance[1] - performance[2])
        if left <= right:
            if performance[0] >= performance[2]:
                # replace the parent in the pop for the kid
                pop[fam[2]] = kids[fam[0]]
            if performance[1] >= performance[3]:
                pop[fam[3]] = kids[fam[1]]
        else:
            if performance[0] >= performance[3]:
                pop[fam[3]] = kids[fam[0]]
                # replace the parent in the pop for the kid
            if performance[1] >= performance[2]:
                pop[fam[2]] = kids[fam[1]]
                # replace the parent in the pop for the kid
    return pop



# instantiate population
pop = np.random.uniform(dom_l, dom_u, (n_pop, n_vars))
for i in range(n_gen):
    print("NEW GENERATION")
    fitness = evaluate(pop)
    kids, family = make_children(pop, fitness, n_pop)
    kids = mutate(kids)
    pop = diversity(kids, pop, family)
#pop,fitness = kill(pop,fitness,kids)


# correlated mutations (CMA-ES)
