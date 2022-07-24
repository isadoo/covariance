#For figure 4
### This script should restart a simulation
## June 29 22

#Using the last iteration of a previous simulation,
#we run a new simulation with no mutation.
#We will track the average fitness of the population for 10 generations. 

import os
import random
import timeit
from datetime import datetime
from functools import reduce
import json
from time import time
from tracemalloc import start
import math
from tempfile import TemporaryFile
import numpy as np
import pandas as pd
import argparse



ITERATIONS     = 10000
AMPLITUDE      = 1
MAP1            = [0.2,0.2,-0.2,-0.2] # cohort 1
MAP2            = [-0.2,0.2,-0.2,0.2] # cohort 2
MAP3            = [0.2,0.2,0.2,0.2]   # cohort 3

# exp 2 is evo_cov_112
# exp 3 is evo_cov_311


STD            = 1
TOTAL_fitcount = []


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--replicate_number", type=int, help="The number of iterations")
parser.add_argument("-mapn", "--map_number", type=int, help="the map to use")


args                         = parser.parse_args()
replicate_N =  args.replicate_number
map_N =  1

if map_N == 1:
    MAP = MAP1
elif map_N == 2:
    MAP = MAP2
elif map_N == 3:
    MAP = MAP3

with open('weights_exp3.npy', 'rb') as f:
    weights = np.load(f, allow_pickle=True)
    weights = np.array(weights)
    print("got weights :", weights.shape)
    
with open('contrib_exp3.npy', 'rb') as f:
    contrib = np.load(f, allow_pickle=True)
    contrib = np.array(contrib)
    print("got contrib :", contrib.shape)

#Fuction takes weights and contribution of a replicate recreates a populaiton
def newevo(w,c):
    phenotypes = []
    fitness    = []

    for i in range(1000):    
       phenotypes.append(np.array(c[i]@w[i].T))

    for i in range(1000):    
       fitness.append(np.array(math.exp(-(sum(phenotypes[i] - MAP)**2)/(2 * STD** 2))))
     
    phenotypes     = np.array(phenotypes)
    total_fitness  = sum(fitness)
    fit_normalized = [*map(lambda x : x/total_fitness, fitness)]
    
    FITCOUNT = []

    for l in range(ITERATIONS):

        FITCOUNT.append(np.mean(fitness))
        indices = list(range(0, len(fitness)))
        birth   = np.random.choice(indices, p = fit_normalized)
        death   = np.random.choice(indices)
        
        w       [death] = w[birth]
        c       [death] = c[birth]
        fitness [death] = fitness[birth]

    return(FITCOUNT)

result_10k =  newevo(weights[replicate_N],contrib[replicate_N])
with open(os.path.join(f"'/Users/idoo/code_covar_evo/cohort{map_N}/", f'replicate{replicate_N}_10k_result.npy'), 'wb') as fw:   
    np.save(fw, np.array(result_10k), allow_pickle = True)
    print(f"Saved 10k of fitness values for replicate \033[093m{replicate_N}\033[0m.")

# pd.DataFrame.from_dict({"fitness":TOTAL_fitcount}).to_csv("fitnesschangeXXX.csv", index=None)
    