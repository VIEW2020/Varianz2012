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
from EvalSurv import EvalSurv
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
    x = data['x']
    time = data['time']
    event = data['event']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df = pd.DataFrame(x, columns=cols_list)
    df['TIME'] = time
    df['EVENT'] = event

    ###################################################################
    
    print('Fitting...')
    cph = CoxPHFitter()
    cph.fit(df, duration_col='TIME', event_col='EVENT', show_progress=True, step_size=0.5)
    cph.print_summary()
    print('done')
        
    ###################################################################
    
    print('Predicting...')
    es = EvalSurv(pd.DataFrame({'LPH': np.dot(x-cph._norm_mean.values, cph.params_), 'TIME': time, 'EVENT': event}))
    df_cox = pd.DataFrame({'LPH': es.df['LPH'], 'RISK_PERC': es.get_risk_perc(1826)})
    df_cox.to_feather(hp.results_dir + 'df_cox_' + hp.gender + '.feather')

if __name__ == '__main__':
    main()
