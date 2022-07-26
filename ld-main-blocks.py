from cmath import log
import os
from pprint import pprint
from statistics import variance
import sys
from typing import List
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import random

verbose = True if len( sys.argv ) > 1 else False
print("Verbose is {}".format(verbose))

def hl(_:str):
    return "\033[95m{}\033[0m".format(_)

def hl1(_:str):
    return "\033[93m{}\033[0m".format(_)

def getrep(path: str, repnum: int = -1) -> pd.DataFrame:
    """Uploaded a replicate to memory"""
    reppath   = os.path.join(path, 'replicate{}.csv'.format(repnum))
    Replicate = pd.read_csv(reppath, index_col=False)
    print("Opened a {} replicate at {}.".format(np.array(Replicate).shape, reppath))
    # print(Replicate)
    return Replicate

def shuffle_weights_t(w_t: np.ndarray) -> np.ndarray:
    """w_t is  Z x N """
    w_t_P = np.copy(w_t)
    for z in range(w_t.shape[1]):
        np.random.shuffle(w_t_P[:, z])
    return w_t_P

def shuffle_contribs_t(contrib_t: np.ndarray) -> np.ndarray:
    """contrib_t is Z x M x N"""
    contrib_t_P   = np.copy(contrib_t)
    [Z, M, N]     = contrib_t.shape
    _             = []
    shuffled_MNxZ = []

    for i in range(M):
        for j in range(N):
            c_ij = contrib_t_P[:, i, j]
            np.random.shuffle(c_ij)
            shuffled_MNxZ.append(c_ij)

    for i in range(Z):
        _.append(np.array(shuffled_MNxZ)[:, i].reshape((M, N)))

    return np.array(_)

def t_W_getall(df: pd.Series) -> np.ndarray:
    """get all weights from a timestamp as 1000x4"""
    return np.array(list(map(lambda e: np.fromstring(str(e), sep=',', dtype=float), df.iloc[2:1002])))

def t_C_getall(df: pd.Series) -> np.ndarray:
    """get all contributions from a timestamp as 1000x4x4"""
    return np.array(list(map(lambda _: np.fromstring(str(_), sep=',', dtype=float).reshape((4, 4)), df.iloc[1002:])))

def covmat_to_avg_tuple__blocks(C, exp_n: int)->List[float]:
    """
    Map contains indices of a given cov. matrix (4x4 or 8x8) over which we ought to take the means.

    @coupled -- accounts for block-correlations
    @rest    -- should   be non-significant on average (given that they are not block-rigged      )
    
    @returns [u_variance, u_cov_coupled, u_cov_rest]
    """
    C = np.array(C)

    PAIRING_MAP = {
        
        31: {
            "variance"   : [[_, _] for _ in range(4) ],
            "cov_coupled": [[0, 1], [2, 3]],
            "cov_rest"   : [[0, 3], [0, 2], [1, 3], [1, 2]]
        },
        32: {
            "variance"   : [[_, _] for _ in range(8)],
            "cov_coupled": [[0, 1], [0, 2], [0, 3],
                            [1, 2], [1, 3],
                            [2, 3],
                            [4, 5], [4, 6], [4, 7],
                            [5, 6], [5, 7],
                            [6, 7]
                            ],
            "cov_rest"   : [
                            [0, 4], [0, 5], [0, 6], [0, 7],
                            [1, 4], [1, 5], [1, 6], [1, 7],
                            [2, 4], [2, 5], [2, 6], [2, 7]
                        ]
        },
        33: {
            "variance"   : [[_, _] for _ in range(8)],
            "cov_coupled": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "cov_rest"   : [
                            [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                            [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
                            [2, 4], [2, 5], [2, 6], [2, 7],
                            [3, 4], [3, 5], [3, 6], [3, 7],
                            [4, 6], [4, 7],
                            [5, 6], [5, 7]
                        ]
        }
    }

    MAP_PICKED = PAIRING_MAP[exp_n]

    return [
                np.mean([C[ind_pair[0],ind_pair[1]] for ind_pair in MAP_PICKED['variance']]),
                np.mean([C[ind_pair[0],ind_pair[1]] for ind_pair in MAP_PICKED['cov_coupled']]),
                np.mean([C[ind_pair[0],ind_pair[1]] for ind_pair in MAP_PICKED['cov_rest']])
            ]

def rep_covs(cor_exp_path, repn: int) -> List:
    """get average(over timestamps ) covariance matrices for replicate: [original, permuted]"""

    r = getrep(cor_exp_path, repnum=repn)

    cov_all   = []
    cov_all_P = []

    for i, Timestamp in enumerate(r.iterrows()):
        # print(" \n------------------| Timestamp {} |---------------------".format(i))

        contribs_t, weights_t = [t_C_getall(
            Timestamp[1]), t_W_getall(Timestamp[1])]

        # print(" --| Got Contribs & Weights  |-- ".format(i))
        # print("contributions :", np.shape(contribs_t))
        # print("weights:"       , np.shape(weights_t ))

        weights_t_P = shuffle_weights_t(weights_t)
        contribs_t_P = shuffle_contribs_t(contribs_t)
        # print(" --| Shuffled Contribs & Weights  {} |-- ".format(i))
        # print("Permuted contributions :", np.shape(contribs_t))
        # print("Permuted weights:"       , np.shape(weights_t ))

        phens_t = np.array([np.array(_[0]@_[1].T)
                           for _ in zip(contribs_t, weights_t)])
        phens_t_P = np.array([np.array(_[0]@_[1].T)
                             for _ in zip(contribs_t_P, weights_t_P)])
        # print(" --| Calculated Phenotypes for Timestamp {} |-- ".format(i))

        cov_all  .append(np.cov(phens_t.T))        # originally, this has a .T
        cov_all_P.append(np.cov(phens_t_P.T))      # originally, this has a .T

    return [
        np.array(cov_all).mean(axis=0),
        np.array(cov_all_P).mean(axis=0)
    ]

# ? ----------------------------------------- CONFIGS ------------------------ |
# If one of these changes -- all likely have to.
do_N_iter            = 500
EXP_N                = 31


EXPERIMENT_DIR_COV   = './x500_replicates/exp3.1.1'
EXPERIMENT_DIR_UNCOV = './x500_replicates/exp1.1.2'

#? Correlated
exp_cor_covs     = []
exp_cor_covs_P   = []

#? Uncorrelated
exp_uncor_covs   = []
exp_uncor_covs_P = []

for _i in range(1, do_N_iter):
    print(
        "\n\n|--------------[Correlated] Replicate #{}---------------|".format(_i))
    try:
        os.path.join(EXPERIMENT_DIR_COV, 'csvs')
        u_cov_cor, u_cov_cor_P = rep_covs(os.path.join(EXPERIMENT_DIR_COV, 'csvs'), _i)


        vanilla  = covmat_to_avg_tuple__blocks(u_cov_cor, EXP_N)
        permuted = covmat_to_avg_tuple__blocks(u_cov_cor_P, EXP_N)

        shuffled_covar_DEBUG = ""
        if vanilla[1] < permuted[1]:
            shuffled_covar_DEBUG = "\t [ X ]" # Flag for permuted covariance being larger than upermuted (suspect).
        print("\tAverage [ VAR,COUPLED,REST ] : [ {},{},{} ] "                .format(*covmat_to_avg_tuple__blocks(u_cov_cor  , EXP_N)                      ))
        print("\tAverage [ VAR,COUPLED,REST ] : [ {},{},{} ] <---Permuted {}" .format(*covmat_to_avg_tuple__blocks(u_cov_cor_P, EXP_N), shuffled_covar_DEBUG))
        exp_cor_covs.append(u_cov_cor)
        exp_cor_covs_P.append(u_cov_cor_P)
    except:
        print("Skipped replicate {}".format(_i))
        ...

for _j in range(1, do_N_iter):
    print(
        "|--------------[Uncorrelated] Replicate #{}---------------|".format(_j))
    try:

        u_cov_uncor, u_cov_uncor_P = rep_covs(os.path.join(EXPERIMENT_DIR_UNCOV, 'csvs'), _j)

        exp_uncor_covs.append(u_cov_uncor)
        exp_uncor_covs_P.append(u_cov_uncor_P)
    except Exception as e:
        print("Skipped replicate {}".format(_j))
        print(e)
        ...


# ----------------- Truncate on lowest number of replicates in case of incongruence
print("""
      

#? Correlated
exp_cor_covs     = {}

#? Uncorrelated
exp_uncor_covs   = {}      
      """.format(exp_cor_covs,
                 exp_uncor_covs))
      



lower_len = len(u_cov_cor) if len(u_cov_cor) < len(u_cov_uncor) else len(u_cov_uncor)

u_cov_cor     = u_cov_cor[:lower_len]
u_cov_uncor   = u_cov_uncor[:lower_len]
u_cov_cor_P   = u_cov_cor_P[:lower_len]
u_cov_uncor_P = u_cov_uncor_P[:lower_len]


delta_cor   = np.array(exp_cor_covs  ) - np.array(exp_cor_covs_P  )
delta_uncor = np.array(exp_uncor_covs) - np.array(exp_uncor_covs_P)

# ---------------------------------------------------------------------------------

#Comparing coupled vs uncoupled vs uncorrelated

#Coupled vs Uncoupled










# ---------------------------------------------------------------------------------

# Standard Error
[ SE__u_cov_cor, 
  SE__u_cov_uncor, 
  SE__u_cov_cor_P,
  SE__u_cov_uncor_P, 
  SE__delta_cor,
  SE__delta_uncor
  ] = [np.std(_) for _ in [u_cov_cor,u_cov_uncor, u_cov_cor_P,u_cov_uncor_P, delta_cor,delta_uncor]]

print(hl("\t\tSTANDARD ERROR"))

print(hl1("\nSE__u_cov_cor"))
pprint(SE__u_cov_cor)
print(hl1("\nSE__u_cov_uncor"))
pprint(SE__u_cov_uncor)
print(hl1("\nSE__u_cov_cor_P"))
pprint(SE__u_cov_cor_P)
print(hl1("\nSE__u_cov_uncor_P"))
pprint(SE__u_cov_uncor_P)
print(hl1("\nSE__delta_cor"))
pprint(SE__delta_cor)
print(hl1("\nSE__delta_uncor"))
pprint(SE__delta_uncor)


# ---------------------------------------------------------------------------------
# Means - Correlated
u1_1 = np.array(exp_cor_covs).mean(axis=0)
u1_2 = np.array(exp_cor_covs_P).mean(axis=0)

# ?  Uncorrelated
u2_1 = np.array(exp_uncor_covs).mean(axis=0)
u2_2 = np.array(exp_uncor_covs_P).mean(axis=0)

# ?  Delta
u_delta_cor   = delta_cor  .mean(axis=0)
u_delta_uncor = delta_uncor.mean(axis=0)

# Round (flip on and off)
[u1_1,u1_2,u2_1,u2_2,u_delta_cor,u_delta_uncor] = [np.round(_,4) for _ in [u1_1,u1_2,u2_1,u2_2,u_delta_cor,u_delta_uncor] ]

print("\n\n\t\t \033[92mCORRELATED\033[0m ")
print("[MEAN VANILLA]")
pprint(np.round(u1_1, 5))
print("(avg) [ VARIANCE, COUPLED_COV, REST_COV ]: ", np.round(covmat_to_avg_tuple__blocks(u1_1, EXP_N), 5))

print("[MEAN PERMUTED]")
pprint(np.round(u1_2, 5))
print("(avg) [ VARIANCE, COUPLED_COV, REST_COV ]: ", np.round(covmat_to_avg_tuple__blocks(u1_2, EXP_N), 5))

print("[DELTA] (Diff between perm and unperm)")
pprint(np.round(u_delta_cor, 5))
print("(avg) [ VARIANCE, COUPLED_COV, REST_COV ]: ", np.round(covmat_to_avg_tuple__blocks(u_delta_cor, EXP_N), 5))


print("\n\n\t\t \033[92mUNCORRELATED\033[0m ")

print("[MEAN VANILLA]")
pprint(np.round(u2_1, 5))
print("(avg) [ VARIANCE, COUPLED_COV, REST_COV ]: ", np.round(covmat_to_avg_tuple__blocks(u2_1, EXP_N), 5))

print("[MEAN PERMUTED]")
pprint(np.round(u2_2, 5))
print("(avg) [ VARIANCE, COUPLED_COV, REST_COV ]: ", np.round(covmat_to_avg_tuple__blocks(u2_2, EXP_N), 5))

print("[DELTA] (Diff between perm and unperm)")
pprint(np.round(u_delta_uncor, 5))
print("(avg) [ VARIANCE, COUPLED_COV, REST_COV ]: ", np.round(covmat_to_avg_tuple__blocks(u_delta_uncor, EXP_N), 5))

print('\nttest(correlated,uncorrelated):')
TT_cor_vs_uncor     = scipy.stats.ttest_ind(exp_cor_covs  , exp_uncor_covs  )
TT_cor_P_vs_uncor_P = scipy.stats.ttest_ind(exp_cor_covs_P, exp_uncor_covs_P)
print("""
TT_cor_vs_uncor     : {}
TT_cor_P_vs_uncor_P : {}      
      """.format(TT_cor_vs_uncor, TT_cor_P_vs_uncor_P))



print('\nttest(Permuted,unmpermputed):')
TT_cor_vs_cor_P     = scipy.stats.ttest_ind(exp_cor_covs  , exp_cor_covs_P  )
TT_uncor_vs_uncor_P = scipy.stats.ttest_ind(exp_uncor_covs, exp_uncor_covs_P)
print("""
TT_cor_vs_cor_P    : {}
TT_uncor_vs_uncor_P: {}      
      """.format(TT_cor_vs_cor_P, TT_uncor_vs_uncor_P))

TT_delta_cor_vs_delta_uncor     = scipy.stats.ttest_ind(delta_cor  , delta_uncor   )
TT_delta_cor_vs_vanilla_cor     = scipy.stats.ttest_ind(delta_cor  , exp_cor_covs  )
TT_delta_uncor_vs_vanilla_uncor = scipy.stats.ttest_ind(delta_uncor, exp_uncor_covs)
print('\nDeltas:')
print("""
TT_delta_cor_vs_delta_uncor    : {}
TT_delta_cor_vs_vanilla_cor    : {}      
TT_delta_uncor_vs_vanilla_uncor: {}      
      """.format(TT_delta_cor_vs_delta_uncor, TT_delta_cor_vs_vanilla_cor, TT_delta_uncor_vs_vanilla_uncor))



