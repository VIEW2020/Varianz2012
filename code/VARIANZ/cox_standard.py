'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import sys
sys.path.append('../lib/')

import numpy as np
import pandas as pd
from deep_survival import *
from utils import *
import feather
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.evaluation import EvalSurv
from lifelines import CoxPHFitter
from hyperparameters import Hyperparameters
from sklearn import preprocessing
import lifelines

from pdb import set_trace as bp


def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
    
    print('Use all data for model fitting...')
    x = np.concatenate((data['x_trn'], data['x_val'], data['x_tst']))
    time = np.concatenate((data['time_trn'], data['time_val'], data['time_tst']))
    event = np.concatenate((data['event_trn'], data['event_val'], data['event_tst']))
    
    x_tst = data['x_tst']
    time_tst = data['time_tst']
    event_tst = data['event_tst']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df = pd.DataFrame(x, columns=cols_list)
    df['TIME'] = time
    df['EVENT'] = event

    ###################################################################
    
    print('Fitting...')
    cph = CoxPHFitter()
    cph.fit(df, duration_col='TIME', event_col='EVENT', show_progress=True, step_size=0.5)
    cph.print_summary()
    #males: 0.9751757392502516, females: 0.988709394816069
    base_surv = baseline_survival(df, np.dot(x-cph._norm_mean.values, cph.params_)).loc[1826]
    print(base_surv)
    print('done')
        
    ###################################################################
    
    print('Predicting...')
    risk = 100*(1-np.power(base_surv, np.exp(np.dot(x_tst-cph._norm_mean.values, cph.params_))))
    df_cox = pd.DataFrame({'RISK': risk})
    
    df_cox.to_feather(hp.results_dir + 'df_cox_' + hp.gender + '.feather')

if __name__ == '__main__':
    main()
