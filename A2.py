# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from itertools import product
from functools import reduce
import time
import matplotlib.pyplot as plt


# import libraries
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random
import pickle as pkl
import multiprocessing
from joblib import Parallel, delayed


# make folder for results
#experiment_name = 'EA2'
#if not os.path.exists(experiment_name):
#    os.makedirs(experiment_name)

# initializes environment with ai player using self-made controller, playing against static enemy
env = Environment(experiment_name='saturday',
                  #enemies=[2,3,4],
                  player_controller=player_controller(),
                  level=2,
                  speed="fastest")


# set time marker
ini = time.time()


# def create_random_rulebook():
#     """Initializes a genotype for one agent, returns this agent a dictionary."""
#     t0 = time.time()
#     rulebook = {}
#     # iterate over possible inputs
#     for i in range(1048575):
#         key = str(bin(i))[2:].zfill(20)
#         outputstring  = list(bin(np.random.randint(31))[2:].zfill(5))
#         # convert to list of ints
#         output = [int(i) for i in outputstring]
#         rulebook[key] = output
#     t1 = time.time()
#     return rulebook


def evaluate(pop):
    """Takes in a list of the population, simulates their performance,
    and returns a list of the fitness scores of each member of the population."""
    fitness = []
    player_live = []

    # simulate game for every agent in the population
    for agent in pop:
        agent_fitness = []
        agent_life = []

        # enemy 1
        env.update_parameter('enemies', [1])
        f,p,e,t = env.play(pcont=agent)
        agent_fitness.append(f)
        agent_life.append(p)

        # enemy 5
        env.update_parameter('enemies', [5])
        f,p,e,t = env.play(pcont=agent)
        # remember fitness
        agent_fitness.append(f)
        agent_life.append(p)

        # get average fitness and player life
        fitness.append(np.average(agent_fitness))
        player_live.append(np.average(agent_life))
    return fitness,player_live


# def crossover(agent_x, agent_y): OLD
#     """Takes in two agents, recombines their genes at a random point,
#     and returns two offspring."""
#     new_agent = {}
#     new_agent_two = {}
#     len_dict = len(list(agent_x.values()))
#     keys = list(agent_x.keys())
#     ind = np.random.randint(1,len_dict)
#     # create first kid
#     other_controls_first = list(agent_x.values())[:ind] + list(agent_y.values())[ind:]
#     for i in range(len_dict):
#         new_agent[keys[i]] = other_controls_first[i]
#     # create second kid
#     ind2 = np.random.randint(1,len_dict)
#     other_controls_second = list(agent_x.values())[:ind] + list(agent_y.values())[ind:]
#     for j in range(len_dict):
#         new_agent_two[keys[j]] = other_controls_second[j]
#     return new_agent, new_agent_two

def crossover(agent_x, agent_y):
     ind = np.random.randint(1,n_vars-2)
     new_agent_first = np.concatenate((agent_x[:ind], agent_y[ind:]), axis=None)
     new_agent_second = np.concatenate((agent_x[ind:], agent_y[:ind]), axis=None)
     return new_agent_first, new_agent_second


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
    n_kids = int(0.2 * n_pop)
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


def mutate(kids,n_flip,mutation):
    """Takes in a list of the offspring, and parameters of the proportion of children to mutate
    and the proportion of mutations to perform on each child. Performs mutations and returns a
    list of the updated offspring"""
    to_mutate = int(mutation*len(kids))
    mutated_kids = []
    # list of keys and possible outputs
    #keys = list(kids[0].keys())
    #outputs = list(product([0,1], repeat=5))
    # to lists
    #outputs = [list(o) for o in outputs]
    for i in range(to_mutate):
        ind = np.random.randint(len(kids)-1)
        while ind in mutated_kids:
            ind = np.random.randint(len(kids)-1)
        for j in range(n_flip):
            toflip = np.random.randint(n_vars)
            kids[ind][toflip] = np.random.uniform(-1,1)
        mutated_kids.append(ind)
    return kids


def crowding_selection(kids, pop, family):
    """Takes in a list of the offspring, the population and the families. Determines
    which individuals of the family survive. Returns the updated population."""
    # get fitnesses of all family members
    for fam in family:
        performance, life = evaluate([kids[fam[0]],kids[fam[1]],pop[fam[2]],pop[fam[3]]])
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


def evolve(n_point_mut,mutation,number_agents,number_gen):
#def evolve(n_point_mut,mutation):
    """Does not take in arguments, runs the evolutionary algorithm and returns the best performing agent,
    a list of the best fitness of each generation and the fitnesses of the complete last generation."""
    # define parameter
    global n_pop, n_gen,n_vars
    #n_point_mut = 50
    n_pop = int(number_agents)
    n_gen = int(number_gen)
    #n_pop = 10
    #n_gen = 2
    #mutation = 0.2
    n_hidden = 10
    n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5
    print("n_vars = ", n_vars)
    t0 = time.time()
    dom_u = 1
    dom_l = -1
    pop = np.random.uniform(dom_l, dom_u, (n_pop, n_vars))
    fitnesses = []
    best_fitness = []
    av_fitnesses = []
    generation_lives = []

    # run algorithms
    for i in range(n_gen):
        #print("NEW GENERATION")
        # evaluate current population
        fitness, player_lives = evaluate(pop)
        fitnesses.append(fitness)
        generation_lives.append(player_lives)
        av_fitnesses.append(np.average(fitness))
        best_fitness.append(max(fitness))
        #print('BEST_FITNESS : ', max(fitness))
        #print('AVERAGE_FITNESS : ',np.average(fitness))
        # reproduce
        kids, family = make_children(pop, fitness, n_pop)
        # mutate  children
        kids = mutate(kids,int(n_point_mut),int(mutation))
        # select new generation
        pop = crowding_selection(kids,pop,family)
    fittest_ind = np.argmax(fitness)
    fittest_value = max(fitness)

    return [fittest_value,pop[fittest_ind],best_fitness,fitnesses,av_fitnesses,generation_lives]


# if __name__ == "__main__":
#
#     pop,best_fitness,fitness = evolve()
#     plt.plot(best_fitness, label = "fittest")
#     plt.plot(fitness, label = "average")
#     plt.legend()
#     plt.show()
#     outfile = "populations/Contrlau_fittest_agent.p"
#     result_out = "results/Contrlau_best_fit1.p"
#     result_out2 = "results/Contlr_fitness1.p"
#     pkl.dump(pop,open(outfile, 'wb'))
#     pkl.dump(best_fitness,open(result_out,'wb'))
#     pkl.dump(fitness,open(result_out2,'wb'))
if __name__ == "__main__":
    m = sys.argv[1]
    m = eval(m.split()[0])
    #frac_mut_list = 0.9#,.3,.9
    #frac_mut_list = m[0]
    #n_to_flip = m[1]

    #n_to_flip = 1000#,10000,50000
    # fitness_list = []
    # lifes_list = []
    # count = 0

    #for i in range(9):
    #t1 = time.time()
    best_sol,best_individ,best,fitnesses,av_fitnesses, gen_lives = evolve(m[1],m[0],m[2],m[3])
    #best_sol,best_individ,best,fitnesses,av_fitnesses, gen_lives = evolve(m[1],m[0])
    print(200 - best_sol)
    #fitness_list.append(fitnesses)
    #lifes_list.append(gen_lives)
    #t2 = time.time()
    #count += 1
    #print("Round : %i"%count,"finished in : %s"%round(t2 - t1,3))

    # outfile = "results/algorithm_diversity_fitness_negen.pkl"
    # pkl.dump(fitness_list,open(outfile,'wb'))
    # outfile = "results/algorithm_diversity_lifes_negen.pkl"
    # pkl.dump(lifes_list,open(outfile,'wb'))
