# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controller_lau1 import controller_automata
from itertools import product
from functools import reduce


# import libraries
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random
import pickle as pkl

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name='saturday',
                  enemies=[2],
                  player_controller=controller_automata(),
                  level=2,
                  speed="fastest")


#env.play()

# set time marker
ini = time.time()



def create_random_rulebook():
    rulebook = {}
    # list of keys : 
    keys = list(product([0,1], repeat=20))
    # possible outputs : 
    outputs = list(product([0,1], repeat=5))

    # convert to lists :
    keys = [reduce(lambda x,y:str(x) + str(y),k) for k in keys]
    outputs = [list(o) for o in outputs]
    # iteratively create dict with random outputs: 
    for key in keys:
        ind = random.randint(0,24)
        rulebook[key] = outputs[ind]
    print('Creating Rulebook')
    return rulebook


def evaluate(pop):
    fitness = []
    for agent in pop:
        f,p,e,t = env.play(pcont=agent)
        fitness.append(f)
    print(f"best fitness = {max(fitness)}")
    return fitness

def crossover(agent_x, agent_y):
    new_agent = {}
    len_dict = len(list(agent_x.values()))
    keys = list(agent_x.keys())
    ind = np.random.randint(1,len_dict)
    other_controls = list(agent_x.values())[:ind] + list(agent_y.values())[ind:]
    for i in range(len_dict):
        new_agent[keys[i]] = other_controls[i]
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
    n_kids = int(0.35 * n_pop)
    for i in range(n_kids):
        while True:
            first = random.uniform(0, cumchance[-1])
            second = random.uniform(0, cumchance[-1])
            for j in range(len(cumchance)-1):
                if first < cumchance[j]:
                    ind1 = i
                    break
            for z in range(len(cumchance)-1):
                if second < cumchance[z]:
                    ind2 = i
                    break
            if (ind1 != ind2):
                break
        kids.append(crossover(pop[ind1], pop[ind2]))
        print(ind1,ind2)
    return kids

def mutate(kids,n_flip):
    to_mutate = int(mutation*len(kids))
    mutated_kids = []
    #list of keys and possible outputes : 
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
            #print('fitnes before mutation : ',evaluate([kids[ind]]))
            kids[ind][keys[toflip]] = outputs[random_out]
            #print('fitness after mutation : ',evaluate([kids[ind]]))
        mutated_kids.append(ind)
    return kids

def kill(pop,fitness,kids):
    fitness, pop = order_to_fitness(fitness,pop)
    kids_fitness = evaluate(kids)
    kids_fitness,kids = order_to_fitness(kids_fitness,kids)
    #print('list of population fitnes ;',fitness)
    #print('list of kids fitness ; ',kids_fitness)
    index = 0
    for i in range(len(kids)):
        for j in range(index,len(pop)):
            if kids_fitness[i] > fitness[j]:
                pop[j] = kids[i]
                fitness[j] = kids_fitness[i]
                index = j
                break

    return pop,fitness


if __name__ == "__main__":
    # params
    n_point_mut = 50
    n_pop = 50
    n_gen = 20
    mutation = 0.5
    n_rules = 1048576

    pop = [create_random_rulebook() for i in range(n_pop)]
    fitnesses = []
    best_fitness = []

    #Evolve
    for i in range(n_gen):
        # evaluate population
        fitness = evaluate(pop)
        fitnesses.append(np.average(fitness))
        best_fitness.append(max(fitness))
        print('BEST_FITNESS : ', max(fitness))
        print('AVERAGE_FITNESS : ',np.average(fitness))
        # reproduce : 
        kids = make_children(pop, fitness, n_pop)
        # mutate :
        kids = mutate(kids,n_point_mut)
        # kill some peeps
        pop,fitness = kill(pop,fitness,kids)
    outfile = "populations/ConL_first_test.p"
    result_out = "results/ConL_best_fit1.p"
    result_out2 = "results/ConL_fitness1.p"
    #pkl.dump(pop,open(outfile, 'wb'))
    pkl.dump(best_fitness,open(result_out,'wb'))
    pkl.dump(fitness,open(result_out2,'wb'))




#env.save_state()

# correlated mutations (CMA-ES)
