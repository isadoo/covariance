#For figure 4
#transform npy into csv 
import random
import timeit
from datetime import datetime
from functools import reduce
import json
from time import time
import sys, os
from tracemalloc import start
from typing import Callable, List
from unicodedata import unidata_version
import numpy as np
import math
import argparse
import pickle
import pandas as pd
from tempfile import TemporaryFile

fitness = []

def getrep(path: str):
    with open(path, 'rb') as f:
        replicate = np.load(f, allow_pickle=True)
        # #print(np.array(weights).shape)
        # replicate = np.array(replicate)
        # print("got rep", replicate)
        return replicate

for _i in range(1,498):
    print("Processed replicate {}".format(_i))
    path_to_replicate = os.path.join('./Figure4/exp3/cohort3/', f'replicate{_i}_10k_result.npy')
    # temporario = getrep(path_to_replicate)
    #LastIterations.append(temporario.iloc[-1])
    # fitness.append(getrep(temporario))
    replicate_from_file = getrep(path_to_replicate)
    fitness.append(replicate_from_file)
    
    
averageFIT = np.mean(fitness, axis=0)

FITdf = pd.DataFrame(averageFIT)

FITdf.to_csv("fitExperiment3Cohort3.csv",index=False)

