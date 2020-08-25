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
    df_cml['LPH'] = feather.read_dataframe(hp.results_dir + 'df_cml_' + hp.gender + '.feather')['LPH']
    
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
    
    # baseline survival
    es_cox = EvalSurv(df_cox.copy())
    es_cml = EvalSurv(df_cml.copy())    
    print('Base survival Cox: {:.5}'.format(es_cox.get_base_surv(1826)))
    print('Base survival CML: {:.5}'.format(es_cml.get_base_surv(1826)))
    return
    
    # evaluate
    for fold in range(hp.num_folds):
        print('Fold: {}'.format(fold))
        
        es_cox = EvalSurv(df_cox.loc[data['fold'] == fold].copy())
        es_cml = EvalSurv(df_cml.loc[data['fold'] == fold].copy())

        r2_vec_cox[fold] = es_cox.R_squared_D()
        d_index_vec_cox[fold], _ = es_cox.D_index()
        concordance_vec_cox[fold] = es_cox.concordance_index()
        ibs_vec_cox[fold] = es_cox.integrated_brier_score()
        auc_vec_cox[fold] = es_cox.auc(1826)
        
        r2_vec_cml[fold] = es_cml.R_squared_D()
        d_index_vec_cml[fold], _ = es_cml.D_index()
        concordance_vec_cml[fold] = es_cml.concordance_index()
        ibs_vec_cml[fold] = es_cml.integrated_brier_score()
        auc_vec_cml[fold] = es_cml.auc(1826)

    r2_mean, (r2_lci, r2_uci) = r2_vec_cox.mean(), sms.DescrStatsW(r2_vec_cox).tconfint_mean()
    print('R-squared(D) Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(r2_mean, r2_lci, r2_uci))
    d_index_mean, (d_index_lci, d_index_uci) = d_index_vec_cox.mean(), sms.DescrStatsW(d_index_vec_cox).tconfint_mean()
    print('D-index Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(d_index_mean, d_index_lci, d_index_uci))
    concordance_mean, (concordance_lci, concordance_uci) = concordance_vec_cox.mean(), sms.DescrStatsW(concordance_vec_cox).tconfint_mean()
    print('Concordance Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(concordance_mean, concordance_lci, concordance_uci))
    ibs_mean, (ibs_lci, ibs_uci) = ibs_vec_cox.mean(), sms.DescrStatsW(ibs_vec_cox).tconfint_mean()
    print('IBS Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(ibs_mean, ibs_lci, ibs_uci))
    auc_mean, (auc_lci, auc_uci) = auc_vec_cox.mean(), sms.DescrStatsW(auc_vec_cox).tconfint_mean()
    print('AUC Cox (95% CI): {:.3} ({:.3}, {:.3})'.format(auc_mean, auc_lci, auc_uci))

    r2_mean, (r2_lci, r2_uci) = r2_vec_cml.mean(), sms.DescrStatsW(r2_vec_cml).tconfint_mean()
    print('R-squared(D) CML (95% CI): {:.3} ({:.3}, {:.3})'.format(r2_mean, r2_lci, r2_uci))
    d_index_mean, (d_index_lci, d_index_uci) = d_index_vec_cml.mean(), sms.DescrStatsW(d_index_vec_cml).tconfint_mean()
    print('D-index CML (95% CI): {:.3} ({:.3}, {:.3})'.format(d_index_mean, d_index_lci, d_index_uci))
    concordance_mean, (concordance_lci, concordance_uci) = concordance_vec_cml.mean(), sms.DescrStatsW(concordance_vec_cml).tconfint_mean()
    print('Concordance CML (95% CI): {:.3} ({:.3}, {:.3})'.format(concordance_mean, concordance_lci, concordance_uci))
    ibs_mean, (ibs_lci, ibs_uci) = ibs_vec_cml.mean(), sms.DescrStatsW(ibs_vec_cml).tconfint_mean()
    print('IBS CML (95% CI): {:.3} ({:.3}, {:.3})'.format(ibs_mean, ibs_lci, ibs_uci))
    auc_mean, (auc_lci, auc_uci) = auc_vec_cml.mean(), sms.DescrStatsW(auc_vec_cml).tconfint_mean()
    print('AUC CML (95% CI): {:.3} ({:.3}, {:.3})'.format(auc_mean, auc_lci, auc_uci))

    
if __name__ == '__main__':
    main()

