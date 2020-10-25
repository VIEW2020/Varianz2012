'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

import numpy as np
import pandas as pd
import feather
import pickle as pkl

from utils import *
from hyperparameters import Hyperparameters
import optuna
from tqdm import tqdm
from lifelines import CoxPHFitter

from pdb import set_trace as bp


def objective(trial, df_trn, df_val):
    penalizer = trial.suggest_loguniform('penalizer', 1e-5, 1e2)
    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
    print(trial.params)
    print('Fitting...')
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df_trn, duration_col='TIME', event_col='EVENT', show_progress=True)
    print('done')
    loglike = cph.score(df_val)
    return loglike


def main():
    pp = Hyperparameters()
    
    print('Load data...')
    data = np.load(pp.data_pp_dir + 'data_arrays_' + pp.gender + '.npz')
    df_index_code = feather.read_dataframe(pp.data_pp_dir + 'df_index_code_' + pp.gender + '.feather')
    df_code_cols = feather.read_dataframe(pp.data_pp_dir + 'df_code_cols_' + pp.gender + '.feather')
    cols_list = load_obj(pp.data_pp_dir + 'cols_list.pkl')

    df = pd.DataFrame(data['x'], columns=cols_list)
    df['TIME'] = data['time']
    df['EVENT'] = data['event']
    df = pd.concat([df, df_code_cols], axis=1)

    idx_trn = (data['fold'][:, 0] != 99)
    df_trn = df[idx_trn]
    idx_val = (data['fold'][:, 0] == 99)
    df_val = df[idx_val]

    print('Begin study...')
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(lambda trial: objective(trial, df_trn, df_val), n_trials=100)
    
    print('Save...')
    save_obj(study, pp.log_dir + 'cel_study_' + pp.gender + '.pkl')


if __name__ == '__main__':
    main()
