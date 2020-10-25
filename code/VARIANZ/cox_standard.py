'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from utils import *
import feather
from hyperparameters import Hyperparameters

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    
    print('Use all data for model fitting...')
    x = data['x']
    time = data['time']
    event = data['event']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df = pd.DataFrame(x, columns=cols_list)
    df['TIME'] = time
    df['EVENT'] = event

    ###################################################################
    
    print('Fitting all data...')
    cph = CoxPHFitter()
    cph.fit(df, duration_col='TIME', event_col='EVENT', show_progress=True, step_size=0.5)
    cph.print_summary()

    print('Saving...')
    df_summary = cph.summary
    df_summary['PREDICTOR'] = cols_list
    df_summary.to_csv(hp.results_dir + 'hr_' + hp.gender + '.csv', index=False)
        
    ###################################################################

    print('Test on each fold (train on swapped)...')
    for fold in range(hp.num_folds):
        for swap in range(2):
            print('Fold: {} Swap: {}'.format(fold, swap))

            idx = (data['fold'][:, fold] == (1-swap))
            x = data['x'][idx]
            time = data['time'][idx]
            event = data['event'][idx]

            df = pd.DataFrame(x, columns=cols_list)
            df['TIME'] = time
            df['EVENT'] = event
            
            print('Fitting all data...')
            cph = CoxPHFitter()
            cph.fit(df, duration_col='TIME', event_col='EVENT', show_progress=True, step_size=0.5)
            print('done')
           
            idx = (data['fold'][:, fold] == swap)
            x = data['x'][idx]
            df_cox = pd.DataFrame({'LPH': np.dot(x-cph._norm_mean.values, cph.params_)})

            print('Saving log proportional hazards for fold...')
            df_cox.to_feather(hp.results_dir + 'df_cox_' + hp.gender + '_fold_' + str(fold) + '_' + str(swap) + '.feather')            
    

if __name__ == '__main__':
    main()
