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
        pkl.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


def baseline_survival(df, log_partial_hazard):
    df['PARTIAL_HAZARD'] = np.exp(log_partial_hazard)
    df = df[['TIME', 'EVENT', 'PARTIAL_HAZARD']]
    df = df.groupby(['TIME']).sum().sort_index(ascending=False)
    df['CUM_PARTIAL_HAZARD'] = df['PARTIAL_HAZARD'].cumsum()
    df = df[df['EVENT']>0]
    df['ALPHA'] = np.exp(-df['EVENT']/df['CUM_PARTIAL_HAZARD'])
    df.sort_index(inplace=True)
    df['S0'] = df['ALPHA'].cumprod()
    return df['S0']


def log(model_name, concordance, brier, nbll, hp):
    df = pd.DataFrame({'model_name': model_name,
                       'np_seed': hp.np_seed,
                       'torch_seed': hp.torch_seed,
                       'min_count': hp.min_count,
                       'nonprop_hazards': hp.nonprop_hazards,
                       'batch_size': hp.batch_size,
                       'max_epochs': hp.max_epochs,
                       'patience': hp.patience,
                       'embedding_dim': hp.embedding_dim,
                       'num_months_hx': hp.num_months_hx,
                       'sample_comp_bh': hp.sample_comp_bh,
                       'concordance': concordance,
                       'brier': brier,
                       'nbll': nbll},
                       index=[0])
    with open(hp.data_dir + 'logfile.csv', 'a', newline='\n') as f:
        df.to_csv(f, mode='a', index=False, header=(not f.tell()))
                       
                       




