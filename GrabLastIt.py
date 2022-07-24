## THIS SCRIPT GRABS THE LAST ITERATION OF A SIMULATION AND GETS IT READY FOR A NEW SIMULATION
##June 29 22

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

outfile = TemporaryFile()
lolo = TemporaryFile()
LastIterations = []
weights_LastIt = [] 
contrib_LastIt = []
optima_LastIt = []

################ -- #################
def t_W_getall(df: pd.Series) -> np.ndarray:    
    """get all weights from a timestamp as 1000x4"""
    return np.array(list(map(lambda e: np.fromstring(str(e), sep=',', dtype=float), df.iloc[-1,2:1002])))

def t_C_getall(df: pd.Series) -> np.ndarray:
    """get all contributions from a timestamp as 1000x4x4"""
    return np.array(list(map(lambda _: np.fromstring(str(_), sep=',', dtype=float).reshape((4, 4)), df.iloc[-1,1002:])))

def t_optima(df: pd.Series) -> np.ndarray:
    """get all optima from a timestamp as 1000x4"""
    return np.array(list(map(lambda b: np.fromstring(str(b), sep=',', dtype=float), df.iloc[-1,1])))
    
def getrep(path: str, repnum: int = -1) -> pd.DataFrame:
    if repnum == -1:
        print("Provide valid replicate number")
        exit(1)
    return pd.read_csv(os.path.join(path, 'replicate{}.csv'.format(repnum)), index_col=False)
#####################################################################################################################

for _i in range(2):
    print("Processed replicate {}".format(_i))
    try:
        temporario = getrep('./exp1.1.1/',_i)
        #LastIterations.append(temporario.iloc[-1])
        weights_LastIt.append(t_W_getall(temporario))
        contrib_LastIt.append(t_C_getall(temporario))
        optima_LastIt.append(t_optima(temporario))
    except:
        ...
        
contrib_LastIt = np.array(contrib_LastIt)
weights_LastIt = np.array(weights_LastIt)
optima_LastIt = np.array(optima_LastIt)


with open('test_weights.npy', 'wb') as fw:   
    np.save(fw, weights_LastIt, allow_pickle = True)
    
with open('test_contrib.npy', 'wb') as fc:   
    np.save(fc, contrib_LastIt, allow_pickle = True)

with open('test_optima.npy', 'wb') as fo:
    np.save(fc, optima_LastIt, allow_pickle = True)
    
    

#print(contrib_LastIt)
#print(np.array(contrib_LastIt))
#pd.DataFrame.from_dict({"weights":weights_LastIt, "contribs":contrib_LastIt}).to_csv("experiment1.csv", index=None)