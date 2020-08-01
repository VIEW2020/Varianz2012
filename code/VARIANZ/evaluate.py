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
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

from hyperparameters import Hyperparameters
from utils import *

from pdb import set_trace as bp

def brier_score(risk, event):
    surv = 1-risk/100
    num_events = event.sum()
    brier = ((surv**2)*event).sum()/num_events
    return brier
    

def main():
    # Load data
    print('Load data...')
    hp = Hyperparameters()
    data = np.load(hp.data_pp_dir + 'data_arrays_' + hp.gender + '.npz')
        
    x = data['x_tst']
    time = data['time_tst']
    event = data['event_tst']
    
    cols_list = load_obj(hp.data_pp_dir + 'cols_list.pkl')
    
    df = pd.DataFrame(x, columns=cols_list)
    df['TIME'] = time
    df['EVENT'] = event
    
    df_cox = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '.feather')
    df_cml = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '.feather')
        
    ################################################################################################
    
    c_index = concordance_index(df['TIME'], -df_cox['RISK'], df['EVENT'])
    print('Concordance Index Cox: {}'.format(c_index))
    c_index = concordance_index(df['TIME'], -df_cml['ENSEMBLE'], df['EVENT'])
    print('Concordance Index ML: {}'.format(c_index))
    
    brier = brier_score(df_cox['RISK'], df['EVENT'])
    print('Brier Score Cox: {}'.format(brier))    
    brier = brier_score(df_cml['ENSEMBLE'], df['EVENT'])
    print('Brier Score ML: {}'.format(brier))

if __name__ == '__main__':
    main()

