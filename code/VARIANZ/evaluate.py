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
from EvalSurv import EvalDstat

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
    
    # c_index = concordance_index(df_cox['TIME'], -df_cox['RISK'], df_cox['EVENT'])
    # print('Concordance Index Cox: {:.3}'.format(c_index))
    # c_index = concordance_index(df_cml['TIME'], -df_cml['RISK'], df_cml['EVENT'])
    # print('Concordance Index ML: {:.3}'.format(c_index))
    
    # df_cox['SURV'] = 1-df_cox['RISK']/100
    # # brier = brier_score(df_cox, 1826)
    # brier = integrated_brier_score(df_cox)
    # print('Brier Score Cox: {}'.format(brier))
    # df_cml['SURV'] = 1-df_cml['RISK']/100
    # # brier = brier_score(df_cml, 1826)
    # brier = integrated_brier_score(df_cml)
    # print('Brier Score ML: {}'.format(brier))
    
    eval_cox = EvalDstat(df_cox['LPH'].values, df_cox['TIME'].values, df_cox['EVENT'].values)
    dindex_cox, lCI_cox, uCI_cox = eval_cox.Dindex()
    r2_cox = eval_cox.R_squared_D()
    print('D-index Cox (95% CI): {:.3}'.format(dindex_cox), ' ({:.3}'.format(lCI_cox), ',  {:.3}'.format(uCI_cox), ")")
    print('R-squared(D) Cox: {:.3}'.format(r2_cox))

    eval_cml = EvalDstat(df_cml['LPH'], df_cml['TIME'], df_cml['EVENT'])
    dindex_cml, lCI_cml, uCI_cml = eval_cml.Dindex()
    r2_cml = eval_cml.R_squared_D()
    print('D-index ML (95% CI): {:.3}'.format(dindex_cml), ' ({:.3}'.format(lCI_cml), ',  {:.3}'.format(uCI_cml), ")")
    print('R-squared(D) ML: {:.3}'.format(r2_cml))
    

if __name__ == '__main__':
    main()

