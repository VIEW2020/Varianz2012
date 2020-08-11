'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import numpy as np
import math
import pandas as pd
import pickle as pkl

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
                       
                       




