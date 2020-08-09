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
import statsmodels.stats.api as sms

from hyperparameters import Hyperparameters
from utils import *
from EvalSurv import EvalSurv

from pdb import set_trace as bp
    

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

    # load log partial hazards
    df_cox['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cox_' + hp.gender + '.feather')['LPH']
    df_cml['LPH'] = 0
    for fold in range(hp.num_folds):
        df_cml.loc[data['fold'] == fold, 'LPH'] = feather.read_dataframe(hp.results_dir + 'df_lph_' + hp.gender + '_fold_' + str(fold) + '.feather')['ENSEMBLE'].values
    
    ################################################################################################
    
    # evaluation vectors
    d_index_vec_cox = np.zeros(hp.num_folds)
    r2_vec_cox = np.zeros(hp.num_folds)
    concordance_vec_cox = np.zeros(hp.num_folds)
    ibs_vec_cox = np.zeros(hp.num_folds)
    auc_vec_cox = np.zeros(hp.num_folds)

    d_index_vec_cml = np.zeros(hp.num_folds)
    r2_vec_cml = np.zeros(hp.num_folds)
    concordance_vec_cml = np.zeros(hp.num_folds)
    ibs_vec_cml = np.zeros(hp.num_folds)
    auc_vec_cml = np.zeros(hp.num_folds)
    
    # evaluate
    for fold in range(hp.num_folds):
        print('Fold: {}'.format(fold))
        
        es_cox = EvalSurv(df_cox.loc[data['fold'] == fold].copy())
        es_cml = EvalSurv(df_cml.loc[data['fold'] == fold].copy())

        d_index_vec_cox[fold], _ = es_cox.D_index()
        r2_vec_cox[fold] = es_cox.R_squared_D()
        concordance_vec_cox[fold] = es_cox.concordance_index()
        ibs_vec_cox[fold] = es_cox.integrated_brier_score()
        auc_vec_cox[fold] = es_cox.auc(1826)
        
        d_index_vec_cml[fold], _ = es_cml.D_index()
        r2_vec_cml[fold] = es_cml.R_squared_D()
        concordance_vec_cml[fold] = es_cml.concordance_index()
        ibs_vec_cml[fold] = es_cml.integrated_brier_score()
        auc_vec_cml[fold] = es_cml.auc(1826)
    
    print('R-squared(D) Cox (95% CI): {:.3}'.format(r2_vec_cox.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(r2_vec_cox).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(r2_vec_cox).tconfint_mean()[1]), ')')
    print('D-index Cox (95% CI): {:.3}'.format(d_index_vec_cox.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(d_index_vec_cox).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(d_index_vec_cox).tconfint_mean()[1]), ')')
    print('Concordance Cox (95% CI): {:.3}'.format(concordance_vec_cox.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(concordance_vec_cox).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(concordance_vec_cox).tconfint_mean()[1]), ')')
    print('IBS Cox (95% CI): {:.3}'.format(ibs_vec_cox.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(ibs_vec_cox).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(ibs_vec_cox).tconfint_mean()[1]), ')')
    print('AUC Cox (95% CI): {:.3}'.format(auc_vec_cox.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(auc_vec_cox).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(auc_vec_cox).tconfint_mean()[1]), ')')

    print('R-squared(D) cml (95% CI): {:.3}'.format(r2_vec_cml.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(r2_vec_cml).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(r2_vec_cml).tconfint_mean()[1]), ')')
    print('D-index cml (95% CI): {:.3}'.format(d_index_vec_cml.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(d_index_vec_cml).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(d_index_vec_cml).tconfint_mean()[1]), ')')
    print('Concordance cml (95% CI): {:.3}'.format(concordance_vec_cml.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(concordance_vec_cml).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(concordance_vec_cml).tconfint_mean()[1]), ')')
    print('IBS cml (95% CI): {:.3}'.format(ibs_vec_cml.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(ibs_vec_cml).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(ibs_vec_cml).tconfint_mean()[1]), ')')
    print('AUC cml (95% CI): {:.3}'.format(auc_vec_cml.mean()), 
          ' ({:.3}'.format(sms.DescrStatsW(auc_vec_cml).tconfint_mean()[0]), 
          ', {:.3}'.format(sms.DescrStatsW(auc_vec_cml).tconfint_mean()[1]), ')')

    
if __name__ == '__main__':
    main()

