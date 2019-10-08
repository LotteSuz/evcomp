# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from controller_lau1 import controller_automata
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
    t0 = time.time()
    rulebook = {}

    for i in range(1048575):
        key = str(bin(i))[2:].zfill(20)
        outputstring  = list(bin(np.random.randint(31))[2:].zfill(5))
        # convert to list of ints
        output = [int(i) for i in outputstring]
        rulebook[key] = output
    t1 = time.time()
    print('created a rulebook in : ',round(t1-t0,3))
    return rulebook


def evaluate(pop):
    fitness = []
    player_live = []
    for agent in pop:
        f,p,e,t = env.play(pcont=agent)
        fitness.append(f)
        player_live.append(p)
    #print(f"best fitness = {max(fitness)}")
    return fitness,player_live

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

    # shift fitness distributeion to possitive values :
    fitness = [i + abs(min(fitness)) for i in fitness]
    # don't normalize anymore
    #chance = [(float(i)-fitness[-1]) / (fitness[0] - fitness[-1]) for i in fitness]
    chance = np.cumsum(fitness)
    n_kids = int(frac_kids * n_pop)
    for i in range(n_kids):
        while True:
            first = random.uniform(0, chance[-1])
            second = random.uniform(0, chance[-1])
            for j in range(len(chance)):
                if first <= chance[j]:
                    ind1 = j
                    #print('Found parent 1')
                    break
            for z in range(len(chance)):
                if second <= chance[z]:
                    ind2 = z
                    #print('Found paretn 2')
                    break
            if (ind1 != ind2):
                #print('Not the same parents')
                break
        kids.append(crossover(pop[ind1], pop[ind2]))
        print(ind1,ind2)
    return kids

def mutate(kids,n_flip,mutation):
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
            random_out = np.random.randint(31)
            #print('fitnes before mutation : ',evaluate([kids[ind]]))
            kids[ind][keys[toflip]] = outputs[random_out]
            #print('fitness after mutation : ',evaluate([kids[ind]]))
        mutated_kids.append(ind)
    return kids

def kill(pop,fitness,kids):
    fitness, pop = order_to_fitness(fitness,pop)
    kids_fitness,x = evaluate(kids)
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

def new_genes(pop):
    n_toreplace = int(.2 * len(pop))
    for i in range(n_toreplace):
        victim = np.random.randint(len(pop)-1)
        pop[victim] = create_random_rulebook()
    return pop


def evolve(n_point_mut,mutation):#n_point_mut,mutation
    # params
    global n_pop, n_gen,n_rules, frac_kids
    frac_kids = .2
    #n_point_mut = 10000
    n_pop = 20
    n_gen = 30
    #mutation = 0.2
    n_rules = 1048575
    t0 = time.time()
    pop = [create_random_rulebook() for i in range(n_pop)]
    fitnesses = []
    best_fitness = []
    av_fitnesses = []
    generation_lives = []
    #Evolve
    for i in range(n_gen):
        # evaluate population
        t0 = time.time()
        fitness,player_lives = evaluate(pop)
        t1 = time.time()
        #print('Evaluated in :',round(t1-t0,3))

        fitnesses.append(fitness)
        generation_lives.append(player_lives)
        av_fitnesses.append(np.average(fitness))
        best_fitness.append(max(fitness))

        #print('BEST_FITNESS : ', max(fitness))
        #print('AVERAGE_FITNESS : ',np.average(fitness))

        # reproduce : 
        t2 = time.time()
        kids = make_children(pop, fitness, n_pop)
        t3 = time.time()

        # mutate :
        kids = mutate(kids,n_point_mut,mutation)
        t4 = time.time()

        # kill some peeps
        pop,fitness = kill(pop,fitness,kids)
        t5 = time.time()
        #random substitution of agents : 
        pop = new_genes(pop)

        t5 = time.time()
        elap_time = round(t5-t0,3)
        #print('Finished round %i'%i, 'in %s'%elap_time)
    fittest_ind = np.argmax(fitness)

    return [pop[fittest_ind],best_fitness,fitnesses,av_fitnesses,generation_lives]


if __name__ == "__main__":
    
    num_cores = multiprocessing.cpu_count()
    frac_mut_list = [.3]#,.3,.9
    n_to_flip = [50000]#,10000,50000
    fitnes_list = []
    live_list = []
    count = 0
    #for frac_mut in frac_mut_list:
        # processed_list = Parallel(n_jobs=num_cores)(delayed(evolve)(toflip,frac_mut) for toflip in n_to_flip)
        # data_dict[(str(frac_mut) + '1000')] = processed_list[0][-2:]
        # data_dict[str(frac_mut) + '10000'] = processed_list[1][-2:]
        # data_dict[str(frac_mut) + '50000'] = processed_list[2][-2:]

    for run in range(10):
        t1 = time.time()
        individ,best,fitnesses,av_fitnesses, gen_lives = evolve(1000,.9)
        fitnes_list.append(fitnesses)
        live_list.append(gen_lives)
        #data_dict[str(frac_mut)+ str(to_flip)] = [fitnesses,gen_lives]
        t2 = time.time()
        #count += 1
        print("Round : %i"%run,"finished in : %s"%round(t2 - t1,3))


    outfile = "results/data_fitness_newagents_10runs2.pkl"
    outfile2 = "results/data_lives_newagents_10runs2.pkl"
    pkl.dump(fitnes_list,open(outfile,'wb'))
    pkl.dump(live_list,open(outfile2,'wb'))
    # pop,best_fitness,fitnesses,av_fitnesses = evolve(10000,.2)
    # plt.plot(best_fitness, label = "fittest")
    # plt.plot(av_fitnesses, label = "average")
    # plt.legend()
    # plt.show()
    # outfile = "populations/Contrlau_fittest_agent_nonewagents.p"
    # result_out = "results/Contrlau_best_fit_pmu10000_rmu02_newagents.p"
    # result_out2 = "results/Contlr_allfitness1_mu10000_nonewagents.p"
    # pkl.dump(pop,open(outfile, 'wb'))
    # pkl.dump(best_fitness,open(result_out,'wb'))
    # pkl.dump(fitnesses,open(result_out2,'wb'))




#env.save_state()

# correlated mutations (CMA-ES)
