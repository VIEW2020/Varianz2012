'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import numpy as np
import math
import pandas as pd
import pickle as pkl
from scipy.stats import f

from pdb import set_trace as bp


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


def log(hp, r2, d_index, concordance, ibs, auc):
    df = pd.DataFrame({'model_name': hp.model_name,
                       'np_seed': hp.np_seed,
                       'torch_seed': hp.torch_seed,
                       'batch_size': hp.batch_size,
                       'max_epochs': hp.max_epochs,
                       'num_months_hx': hp.num_months_hx,
                       'r2': r2,
                       'd_index': d_index,
                       'concordance': concordance,
                       'ibs': ibs,
                       'auc': auc},
                       index=[0])
    with open(hp.data_dir + 'logfile.csv', 'a', newline='\n') as f:
        df.to_csv(f, mode='a', index=False, header=(not f.tell()))
                       
                       
def robust_cv_test(res_a, res_b):
    # Combined 5x2cv F Test for Comparing SupervisedClassification Learning Algorithms
    # https://www.cmpe.boun.edu.tr/~ethem/files/papers/NC110804.PDF
    # res_a and res_b are the results of two classifiers with shape num_folds x 2
    assert res_a.shape == res_b.shape, 'The two arrays should have equal dimensions'
    assert res_a.shape[1] == 2, 'Dimension 1 should be 2 for both arrays'
    num_folds = res_a.shape[0]
    
    diff = res_a - res_b
    diff_fold = diff.mean(axis=1, keepdims=True)
    var = ((diff - diff_fold)**2).sum(axis=1)
    f_val = (diff**2).sum()/(2*var.sum())
    p_val = f.sf(f_val, 2*num_folds, num_folds)
    
    return p_val
    
    



