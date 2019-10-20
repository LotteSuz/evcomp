# imports framework
import sys
import os
sys.path.insert(0, 'evoman')
from controller_lau1 import controller_automata
from environment import Environment
from functools import reduce
from itertools import product
import matplotlib.pyplot as plt
import time


# import libraries
from math import fabs, sqrt
import glob, os
import numpy as np
import pickle as pkl
import random
import time


# make folder for results
experiment_name = 'EA2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using self-made controller, playing against static enemy
env = Environment(experiment_name='saturday',
                  enemies=[2],
                  player_controller=controller_automata(),
                  level=2,
                  speed="fastest")


# set time marker
ini = time.time()


def create_random_rulebook():
    """Initializes a genotype for one agent, returns this agent a dictionary."""
    t0 = time.time()
    rulebook = {}
    # iterate over possible inputs
    for i in range(1048575):
        key = str(bin(i))[2:].zfill(20)
        outputstring  = list(bin(np.random.randint(31))[2:].zfill(5))
        # convert to list of ints
        output = [int(i) for i in outputstring]
        rulebook[key] = output
    t1 = time.time()
    return rulebook


def evaluate(pop):
    """Takes in a list of the population, simulates their performance,
    and returns a list of the fitness scores of each member of the population."""
    fitness = []
    # simulate game for every agent in the population
    for agent in pop:
        f,p,e,t = env.play(pcont=agent)
        # remember fitness
        fitness.append(f)
    return fitness


def crossover(agent_x, agent_y):
    """Takes in two agents, recombines their genes at a random point,
    and returns two offspring."""
    new_agent = {}
    new_agent_two = {}
    len_dict = len(list(agent_x.values()))
    keys = list(agent_x.keys())
    ind = np.random.randint(1,len_dict)
    # create first kid
    other_controls_first = list(agent_x.values())[:ind] + list(agent_y.values())[ind:]
    for i in range(len_dict):
        new_agent[keys[i]] = other_controls_first[i]
    # create second kid
    ind2 = np.random.randint(1,len_dict)
    other_controls_second = list(agent_x.values())[:ind] + list(agent_y.values())[ind:]
    for j in range(len_dict):
        new_agent_two[keys[j]] = other_controls_second[j]
    return new_agent, new_agent_two


def order_to_fitness(fitness,pop):
    """Takes in a list of the fitness of all agents in the population, and a list of the population itself,
    orders both lists from high to low based on the fitnesses."""
    fitness,pop = zip(*sorted(zip(fitness,pop),key = lambda fit:fit[0],reverse = True))
    return list(fitness),list(pop)


def make_children(pop, fitness, n_pop):
    """Takes in a list of the population, their fitnesses and the population size, creates offspring,
    and returns a list of this offspring and a list families consisting of two parents and the two children
    they created."""
    kids = []
    family = []
    # order lists from high to low fitness
    fitness,pop = order_to_fitness(fitness,pop)

    # shift fitness distribution to possitive values only
    fitness = [i + abs(min(fitness)) for i in fitness]
    chance = np.cumsum(fitness)
    n_kids = int(0.35 * n_pop)
    # count indeces of kidlist
    c0 = 0
    c1 = 1
    # do this for the amount of kids you want to make
    for i in range(n_kids):
        while True:
            first = random.uniform(0, chance[-1])
            second = random.uniform(0, chance[-1])
            # find first parent
            for j in range(len(chance)):
                if first <= chance[j]:
                    ind1 = j
                    break
            # find second parent
            for z in range(len(chance)):
                if second <= chance[z]:
                    ind2 = z
                    break
            # make sure parents are 2 different agents
            if (ind1 != ind2):
                break
        # make two kids
        first, second = crossover(pop[ind1], pop[ind2])
        kids.append(first)
        kids.append(second)
        family.append([c0,c1,ind1,ind2])
        c0 += 2
        c1 += 2
    return kids, family


def mutate(kids,n_flip):
    """Takes in a list of the offspring, and parameters of the proportion of children to mutate
    and the proportion of mutations to perform on each child. Performs mutations and returns a
    list of the updated offspring"""
    to_mutate = int(mutation*len(kids))
    mutated_kids = []
    # list of keys and possible outputs
    keys = list(kids[0].keys())
    outputs = list(product([0,1], repeat=5))
    # to lists
    outputs = [list(o) for o in outputs]
    for i in range(to_mutate):
        ind = np.random.randint(len(kids)-1)
        while ind in mutated_kids:
            ind = np.random.randint(len(kids)-1)
        for j in range(n_flip):
            toflip = np.random.randint(n_rules)
            random_out = np.random.randint(24)
            kids[ind][keys[toflip]] = outputs[random_out]
        mutated_kids.append(ind)
    return kids


def crowding_selection(kids, pop, family):
    """Takes in a list of the offspring, the population and the families. Determines
    which individuals of the family survive. Returns the updated population."""
    # get fitnesses of all family members
    for fam in family:
        performance = evaluate([kids[fam[0]],kids[fam[1]],pop[fam[2]],pop[fam[3]]])
        # compute intercompetition distances
        left = abs(performance[0] - performance[2]) + abs(performance[1] - performance[3])
        right = abs(performance[0] - performance[3]) + abs(performance[1] - performance[2])
        # replace parent in the population for kid, if the kid's fitness score is higher
        if left <= right:
            if performance[0] >= performance[2]:
                pop[fam[2]] = kids[fam[0]]
            if performance[1] >= performance[3]:
                pop[fam[3]] = kids[fam[1]]
        else:
            if performance[0] >= performance[3]:
                pop[fam[3]] = kids[fam[0]]
            if performance[1] >= performance[2]:
                pop[fam[2]] = kids[fam[1]]
    return pop


def evolve():
    """Does not take in arguments, runs the evolutionary algorithm and returns the best performing agent,
    a list of the best fitness of each generation and the fitnesses of the complete last generation."""
    # define parameters
    global n_point_mut, n_pop, n_gen,mutation,n_rules
    n_point_mut = 124
    n_pop = 16
    n_gen = 38
    mutation = 0.3
    n_rules = 1048576
    t0 = time.time()
    pop = [create_random_rulebook() for i in range(n_pop)]
    fitnesses = []
    best_fitness = []

    # run algorithms
    for i in range(n_gen):
        # evaluate current population
        fitness = evaluate(pop)
        fitnesses.append(np.average(fitness))
        best_fitness.append(max(fitness))
        print('BEST_FITNESS : ', max(fitness))
        print('AVERAGE_FITNESS : ',np.average(fitness))
        # reproduce
        kids, family = make_children(pop, fitness, n_pop)
        # mutate  children
        kids = mutate(kids,n_point_mut)
        # select new generation
        pop = crowding_selection(kids,pop,family)
    fittest_ind = np.argmax(fitness)

    return pop[fittest_ind],best_fitness,fitness


if __name__ == "__main__":

    pop,best_fitness,fitness = evolve()
    plt.plot(best_fitness, label = "fittest")
    plt.plot(fitness, label = "average")
    plt.legend()
    plt.show()
    outfile = "populations/Contrlau_fittest_agent.p"
    result_out = "results/Contrlau_best_fit1.p"
    result_out2 = "results/Contlr_fitness1.p"
    pkl.dump(pop,open(outfile, 'wb'))
    pkl.dump(best_fitness,open(result_out,'wb'))
    pkl.dump(fitness,open(result_out2,'wb'))
