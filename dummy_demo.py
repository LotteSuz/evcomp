################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment

# import libraries
import time
import numpy as np
from math import fabs,sqrt
import glob, os

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
env.play()

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
