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
    
    data = np.load(hp.results_dir + 'risk_cox_standard_' + hp.gender + '.npz')
    df['RISK_COX'] = data['risk']
    
    data = np.load(hp.results_dir + 'risk_matrix_' + hp.gender + '.npz')
    risk_matrix = data['risk_matrix']
    df['RISK_CML'] = risk_matrix.mean(axis=1)
        
    ################################################################################################
    
    c_index = concordance_index(df['TIME'], -df['RISK_COX'], df['EVENT'])
    print('Concordance Index Cox: {}'.format(c_index))
    c_index = concordance_index(df['TIME'], -df['RISK_CML'], df['EVENT'])
    print('Concordance Index ML: {}'.format(c_index))
    brier = brier_score(df['RISK_COX'], df['EVENT'])
    print('Brier Score Cox: {}'.format(brier))    
    brier = brier_score(df['RISK_CML'], df['EVENT'])
    print('Brier Score ML: {}'.format(brier))
    

if __name__ == '__main__':
    main()

