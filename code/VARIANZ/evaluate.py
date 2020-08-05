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
from EvalSurv import EvalSurv

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
    time = data['time']
    event = data['event']
    
    df = pd.DataFrame({'TIME': data['time'], 'EVENT': data['event']})
    df_cox = df.copy()
    df_cml = df.copy()
    
    # load predicted risk
    df_cox['RISK'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '.feather')['RISK']
    df_cml['RISK'] = 0
    for fold in range(hp.num_folds):
        df_cml.loc[data['fold'] == fold, 'RISK'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '_fold_' + str(fold) + '.feather')['ENSEMBLE'].values

    # load log partial hazards
    df_cox['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '.feather')['LPH']
    df_cml['LPH'] = 0
    for fold in range(hp.num_folds):
        df_cml.loc[data['fold'] == fold, 'LPH'] = feather.read_dataframe(hp.results_dir + 'df_lph_' + hp.gender + '_fold_' + str(fold) + '.feather')['ENSEMBLE'].values
    
    # remove validation data
    df_cox = df_cox[data['fold'] != 99]
    df_cml = df_cml[data['fold'] != 99]

    ################################################################################################

    base_surv_cox = baseline_survival(df_cox[['TIME', 'EVENT']], df_cox['LPH'])
    eval_cox = EvalSurv(df_cox, base_surv_cox)
    d_index_cox, lCI_cox, uCI_cox = eval_cox.D_index()
    print('D-index Cox (95% CI): {:.5}'.format(d_index_cox), ' ({:.5}'.format(lCI_cox), ',  {:.5}'.format(uCI_cox), ")")
    print('R-squared(D) Cox: {:.5}'.format(eval_cox.R_squared_D()))
    print('Concordance Cox: {:.5}'.format(eval_cox.concordance_index()))
    print('Brier: {:.5}'.format(eval_cox.brier_score(1826)))
    print('IBS: {:.5}'.format(eval_cox.integrated_brier_score()))

    base_surv_cml = baseline_survival(df_cml[['TIME', 'EVENT']], df_cml['LPH'])
    eval_cml = EvalSurv(df_cml, base_surv_cml)
    d_index_cml, lCI_cml, uCI_cml = eval_cml.D_index()
    print('D-index ML (95% CI): {:.5}'.format(d_index_cml), ' ({:.5}'.format(lCI_cml), ',  {:.5}'.format(uCI_cml), ")")
    print('R-squared(D) ML: {:.5}'.format(eval_cml.R_squared_D()))
    print('Concordance ML: {:.5}'.format(eval_cml.concordance_index()))
    print('Brier: {:.5}'.format(eval_cml.brier_score(1826)))
    print('IBS: {:.5}'.format(eval_cox.integrated_brier_score()))
    
    

    

if __name__ == '__main__':
    main()

