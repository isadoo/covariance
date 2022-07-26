#For figure 4
### This script should restart a simulation
## June 29 22

#Using the last iteration of a previous simulation,
#we run a new simulation with no mutation.
#We will track the average fitness of the population for 10 generations. 

#For figure 4
### This script should restart a simulation
## June 29 22

#Using the last iteration of a previous simulation,
#we run a new simulation with no mutation.
#We will track the average fitness of the population for 10 generations. 

import random
import timeit
from datetime import datetime
from functools import reduce
import json
from time import time
import  os
from tracemalloc import start
import numpy as np
import math
import argparse
import pandas as pd

ITERATIONS     = 10000
AMPLITUDE      = 1
STD            = 1
TOTAL_fitcount = []
# ADDMAP1 = [0.2, 0.2, -0.2, -0.2]
# ADDMAP2 = [-0.2, 0 , 0.2 , -0.2] 
# ADDMAP3 = [ 0.2, 0.2, 0.2, 0.2] 

# ADDMAP1 = [0,0,0,0] # cohort 1
# # ADDMAP2 = [0,0,0,0] # cohort 2
# ADDMAP2 = [-1, 0 , 1 , -1] # cohort 2
# ADDMAP3 = [ 0.2, 0.2, 0.2, 0.2] # cohort 3
# ADDMAP3 = [5,5,5,5] # cohort 3
# ADDMAP3 = [5,5,5,5] # cohort 3
ADDMAP1 = [0.2,1,-0.2,-1]
ADDMAP2 = [-1,0.2,1,-0.2]
ADDMAP3 = [1,1,1,1]

parser = argparse.ArgumentParser()
parser.add_argument("-r"       , "--replicate_number"  , type=int, help="The number of iterations")
parser.add_argument("-addmapn",  "--addmap_number"     , type=int, help="the map to use"          )
parser.add_argument("-exp"     ,"--experiment" ,        type=int, help="exp n"                                 )

args              = parser.parse_args()
replicate_N       = args.replicate_number
addmap_N          = args.addmap_number
EXPERIMENT_NUMBER = args.experiment

if addmap_N == 1:
    ADDMAP = ADDMAP1
elif addmap_N == 2:
    ADDMAP = ADDMAP2
elif addmap_N == 3:
    ADDMAP = ADDMAP3
else:
    print("Error: addmap_N not valid")
    exit()

with open(f'/home/rxz/dev/polygen/test_optima{EXPERIMENT_NUMBER}.npy', 'rb') as f:
    optima = np.load(f, allow_pickle=True)
    optima = np.array(optima)

with open(f'/Users/idoo/code_covar_evo/test_weights{EXPERIMENT_NUMBER}.npy', 'rb') as f:
    weights = np.load(f, allow_pickle=True)
    weights = np.array(weights)
    
with open(f'/Users/idoo/code_covar_evo/test_contrib{EXPERIMENT_NUMBER}.npy', 'rb') as f:
    contrib = np.load(f, allow_pickle=True)
    contrib = np.array(contrib)

   
# print(">>>>>>>>>>>>>>> Chose map number:{} : {} ".format(addmap_N, ADDMAP))
# print(">>>>>>>>>>>>>>> Chose experiment number:{}".format(EXPERIMENT_NUMBER ))
def newevo(w,c,o):
    
    print("\nAdding map {} to optima pre : {}".format(ADDMAP,o))
    MAP = o+ ADDMAP
    print("Map post: {}".format(MAP))
    # print("got MAP on newevo entry: ", MAP)

    phenotypes = []
    fitness    = []

    for i in range(1000):    
       phenotypes.append(np.array(c[i]@w[i].T))

    for i in range(1000):    
       fitness.append(np.array(math.exp(-(sum((phenotypes[i] - MAP)**2)/(2 * STD** 2)))))
     

    #  FITNESS math.exp(-(sum(phenotypes[i] - MAP)**2)/(2 * STD** 2))
    # e to the power of (-( ith phenotype - MAP )^2  divided by double the standard dev squared )
    phenotypes     = np.array(phenotypes)
    total_fitness  = sum(fitness)
    fit_normalized = [*map(lambda x : x/total_fitness, fitness)]
    
    FITCOUNT = []

    for _ in range(ITERATIONS):

        FITCOUNT.append(np.mean(fitness))
        indices = list(range(0, len(fitness)))
        birth   = np.random.choice(indices, p = fit_normalized)
        death   = np.random.choice(indices)
        
        w       [death] = w[birth]
        c       [death] = c[birth]
        fitness [death] = fitness[birth]

    print("{}" .format(FITCOUNT[0])) 
    # print("Mean at newevo end: \t" ,np.mean(FITCOUNT) )
    return(FITCOUNT)

result_10k =  newevo(
                     weights[replicate_N],
                     contrib[replicate_N],
                     optima [replicate_N]
                     )
with open(os.path.join(
    f"/Users/idoo/code_covar_evo/exp{EXPERIMENT_NUMBER}/cohort{addmap_N}/",
    f'replicate{replicate_N}_10k_result.npy'
    ), 'wb') as fw:   
    np.save(fw, np.array(result_10k), allow_pickle = True)
    # print(f"Saved 10k of fitness values for replicate \033[093m{replicate_N}\033[0m.")
    

