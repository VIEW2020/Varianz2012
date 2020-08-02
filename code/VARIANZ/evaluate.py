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
from lifelines import KaplanMeierFitter
import tqdm

from hyperparameters import Hyperparameters
from utils import *

from pdb import set_trace as bp


def get_cens_surv(df):
    # inverse KMF
    kmf = KaplanMeierFitter()
    kmf.fit(df['TIME'], event_observed=(1-df['EVENT']))
    cens_surv = kmf.survival_function_.reset_index()
    cens_surv.rename(columns={'timeline': 'TIME', 'KM_estimate': 'CENS_SURV'}, inplace=True)
    return cens_surv


def brier(df, cens_surv, at_time):
    cens_surv_at_time = cens_surv[cens_surv['TIME'] <= at_time].CENS_SURV.values[-1]
    df['BRIER_1'] = ((df['SURV']**2) * df['EVENT'] * (df['TIME'] <= at_time).astype(int))/df['CENS_SURV']
    df['BRIER_2'] = (((1-df['SURV'])**2) * (df['TIME'] > at_time).astype(int))/cens_surv_at_time    
    brier = (df['BRIER_1'].sum() + df['BRIER_2'].sum())/len(df.index)
    return brier


def brier_score(df, at_time): 
    cens_surv = get_cens_surv(df)
    df = df.merge(cens_surv, how='left', on='TIME')
    return brier(df, cens_surv, at_time)


def integrated_brier_score(df):
    cens_surv = get_cens_surv(df)
    df = df.merge(cens_surv, how='left', on='TIME')
    cens_surv['BRIER'] = 0.
    
    for index, row in cens_surv.iterrows():
        if (index != 0):
            at_time = row['TIME']
            cens_surv.at[index, 'BRIER'] = brier(df, cens_surv, at_time)

    # integrate
    ibs = (cens_surv['TIME'].diff()*cens_surv['BRIER']).sum()/cens_surv['TIME'].max()
    return ibs
    

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
    
    df['SURV'] = 1-df_cox['RISK']/100
    brier = brier_score(df, 1826)
    print('Brier Score Cox: {}'.format(brier))
    df['SURV'] = 1-df_cml['ENSEMBLE']/100
    brier = brier_score(df, 1826)
    print('Brier Score ML: {}'.format(brier))
    

if __name__ == '__main__':
    main()

